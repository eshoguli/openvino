// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {
// namespace aarch64

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

template <cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                     const std::vector<Eltwise::EltwiseData>& eltwise_data,
                                     const std::vector<ov::intel_cpu::Type>& ops_list,
                                     const dnnl::post_ops& post_ops)
    : jit_uni_eltwise_kernel(jep), jit_generator(jit_name()), eltwise_data_(eltwise_data), ops_list_(ops_list), post_ops_(post_ops) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);

        eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
        for (size_t i = 1; i < eltwise_data_.size(); ++i) {
            post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
        }

        const auto& p = post_ops_.get();
        for (int i = 0; i < post_ops_.len(); ++i) {
            if (!p->entry_[i].is_quantization()) {
                IE_THROW() << "Eltwise jitter error. Unsupported post op detected";
            }
            quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                    this, p->entry_[i], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
        }

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, isa));

        const auto &jep = jep_;

        this->preamble();

        const int offset_count = jep.input_size - 1;

        // ptrs initializing
        if (jep.use_runtime_ptrs) {
            for (size_t i = 0; i < jep.inputs_number; i++) {
                mov(start_to_offsets, ptr[reg_const_params + GET_OFF(src_offsets) + i * sizeof(size_t)]);
                mov(get_src_reg(i), ptr[reg_const_params + GET_OFF(src_ptr[0]) + i * sizeof(size_t)]);
                for (int j = 0; j < offset_count; j++) {
                    mov(reg_tmp_64, ptr[start_to_offsets + j * sizeof(size_t)]);
                    imul(reg_tmp_64, ptr[reg_indexes + j * sizeof(size_t)]);
                    add(get_src_reg(i), reg_tmp_64);
                }
            }

            mov(start_to_offsets, ptr[reg_const_params + GET_OFF(dst_offsets)]);
            mov(reg_dst, ptr[reg_const_params + GET_OFF(dst_ptr)]);
            for (int j = 0; j < offset_count; j++) {
                mov(reg_tmp_64, ptr[start_to_offsets + j * sizeof(size_t)]);
                imul(reg_tmp_64, ptr[reg_indexes + j * sizeof(size_t)]);
                add(reg_dst, reg_tmp_64);
            }

            xor_(reg_oc_off, reg_oc_off);

            mov(reg_work_amount, ptr[reg_const_params + GET_OFF(work_amount)]);
        } else {
            auto init_ptrs_with_offsets = [this, offset_count](Reg64 pointer, const std::vector<size_t>& offsets) {
                for (int j = 0; j < offset_count; j++) {
                    if (jep_.dims[j] != 1 && offsets[j] != 0) {
                        mov(reg_tmp_64, offsets[j]);
                        imul(reg_tmp_64, ptr[reg_indexes + j * sizeof(size_t)]);
                        add(pointer, reg_tmp_64);
                    }
                }
            };

            for (size_t i = 0; i < jep.inputs_number; i++) {
                mov(get_src_reg(i), ptr[reg_const_params + GET_OFF(src_ptr[0]) + i * sizeof(size_t)]);
                init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
            }

            mov(reg_dst, ptr[reg_const_params + GET_OFF(dst_ptr)]);
            init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

            xor_(reg_oc_off, reg_oc_off);
            init_ptrs_with_offsets(reg_oc_off, jep.oc_offsets);

            mov(reg_work_amount, jep.work_amount);
        }

        mov(reg_post_op_ptrs, ptr[reg_const_params + GET_OFF(post_op_data)]);

        Xbyak::Label unroll_loop_label;
        Xbyak::Label unroll_loop_end_label;
        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        if (isa == x64::avx512_core)
            vpxord(vmm_zero, vmm_zero, vmm_zero);

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] == 1)
                load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, true);
        }

        size_t min_src_size = jep.dst_size;
        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1)
                min_src_size = std::min(min_src_size, jep.src_size[i]);
        }
        if (jep_.oc_size > 1)
            min_src_size = std::min(min_src_size, jep_.oc_size);

        if (min_src_size != jep.dst_size) {
            bool is_valid_configuration = true;
            if (jep.dst_size % min_src_size != 0)
                is_valid_configuration = false;

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                    is_valid_configuration = false;
            }

            if (jep_.oc_size > 1 && jep_.oc_size != min_src_size && jep_.oc_size != jep.dst_size)
                is_valid_configuration = false;

            if (!is_valid_configuration)
                IE_THROW() << "Eltwise jitter has invalid configuration for Eltwise node";

            L(unroll_loop_label);
            {
                size_t loop_step = min_src_size;
                size_t vec_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(unroll_loop_end_label, T_NEAR);

                for (size_t j = 0; j < min_src_size / vec_step; j++) {
                    for (size_t i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_vector(get_vmm_reg(i), ptr[get_src_reg(i) + j * vec_step * jep.src_prc[i].size()], jep.src_prc[i], exec_prc, false);
                    }

                    compute_eltwise_op();

                    apply_post_ops(false, jep_.oc_size > 1 ? j * vec_step * sizeof(float) : 0);

                    store_vector(ptr[reg_dst + j * vec_step * jep.dst_prc.size()], vmm_dst, exec_prc, jep.dst_prc);
                }

                size_t tail_start = min_src_size - min_src_size % vec_step;
                for (size_t j = tail_start; j < min_src_size; j++) {
                    for (size_t i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_scalar(get_xmm_reg(i), ptr[get_src_reg(i) + j * jep.src_prc[i].size()], jep.src_prc[i], exec_prc);
                    }

                    compute_eltwise_op();

                    apply_post_ops(true, jep_.oc_size > 1 ? j * sizeof(float) : 0);

                    store_scalar(ptr[reg_dst + j * jep.dst_prc.size()], xmm_dst, exec_prc, jep.dst_prc);
                }

                for (size_t i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] == jep.dst_size)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1 && jep_.oc_size != min_src_size)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(unroll_loop_label, T_NEAR);
            }

            L(unroll_loop_end_label);
        }

        if (min_src_size == jep.dst_size) {
            L(main_loop_label);
            {
                size_t loop_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(main_loop_end_label, T_NEAR);

                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1)
                        load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, false);
                }

                compute_eltwise_op();

                apply_post_ops(false);

                store_vector(ptr[reg_dst], vmm_dst, exec_prc, jep.dst_prc);

                for (size_t i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] != 1)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(main_loop_label, T_NEAR);
            }

            L(main_loop_end_label);
        }

        L(tail_loop_label);
        {
            size_t loop_step = 1;

            cmp(reg_work_amount, loop_step);
            jl(tail_loop_end_label, T_NEAR);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1)
                    load_scalar(get_xmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc);
            }

            compute_eltwise_op();

            apply_post_ops(true);

            store_scalar(ptr[reg_dst], xmm_dst, exec_prc, jep.dst_prc);

            for (size_t i = 0; i < jep.inputs_number; i++)
                if (jep.src_size[i] != 1)
                    add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

            add(reg_dst, jep.dst_prc.size() * loop_step);
            sub(reg_work_amount, loop_step);
            if (jep_.oc_size > 1)
                add(reg_oc_off, loop_step * sizeof(float));

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_end_label);

        this->postamble();

        if (uni_vcvtneps2bf16)
            uni_vcvtneps2bf16->emit_data();

        eltwise_emitter->emit_data();
        for (size_t i = 0; i < post_op_emitters.size(); i++) {
            post_op_emitters[i]->emit_data();
        }
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;

    Reg64 get_src_reg(int idx) {
        return Reg64(r8.getIdx() + idx);
    }

    Vmm get_vmm_reg(int idx) {
        return Vmm(1 + idx);
    }

    Vmm get_aux_vmm(int idx) {
        return Vmm(10 + idx);
    }

    Xmm get_xmm_reg(int idx) {
        return Xmm(get_vmm_reg(idx).getIdx());
    }

    Reg64 reg_post_op_ptrs = rax;
    Reg64 start_to_offsets = reg_post_op_ptrs; // rax
    Reg64 reg_dst = rbx;
    Reg64 reg_work_amount = rdx;

    Reg64 reg_oc_off = abi_not_param1;
    Reg64 reg_const_params = abi_param1;
    Reg64 reg_indexes = abi_param2;  // reg_d_bias

    Reg8 reg_tmp_8 = Reg8(r15.getIdx());
    Reg16 reg_tmp_16 = Reg16(r15.getIdx());
    Reg32 reg_tmp_32 = Reg32(r15.getIdx());
    Reg64 reg_tmp_64 = Reg64(r15.getIdx());

    Reg64 reg_d_weights = rbp;
    Reg64 reg_d_bias = rsi;

    Vmm vmm_dst = Vmm(9);
    Xmm xmm_dst = Xmm(9);

    Vmm vmm_d_weights = Vmm(12);
    Vmm vmm_d_bias = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};

    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors = {};

    const std::vector<Eltwise::EltwiseData>& eltwise_data_;
    const std::vector<ov::intel_cpu::Type>& ops_list_;
    const dnnl::post_ops& post_ops_;

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const Eltwise::EltwiseData& data, Precision exec_prec) {
        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            data,
            exec_prec
        };

        OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
        OV_CASE(Algorithm::EltwiseRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluErf, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseElu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAbs, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSqrt, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSoftRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseExp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseClamp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSwish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHswish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseMish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHsigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfToEven, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfAwayFromZero, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
        OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
        OV_CASE(Algorithm::EltwiseFloorMod, jit_floor_mod_emitter),
        OV_CASE(Algorithm::EltwiseMod, jit_mod_emitter),
        OV_CASE(Algorithm::EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(Algorithm::EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(Algorithm::EltwiseSquaredDifference, jit_squared_difference_emitter),
        OV_CASE(Algorithm::EltwisePowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Algorithm::EltwiseEqual, jit_equal_emitter),
        OV_CASE(Algorithm::EltwiseNotEqual, jit_not_equal_emitter),
        OV_CASE(Algorithm::EltwiseGreater, jit_greater_emitter),
        OV_CASE(Algorithm::EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Algorithm::EltwiseLess, jit_less_emitter),
        OV_CASE(Algorithm::EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(Algorithm::EltwiseLogicalAnd, jit_logical_and_emitter),
        OV_CASE(Algorithm::EltwiseLogicalOr, jit_logical_or_emitter),
        OV_CASE(Algorithm::EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(Algorithm::EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
        OV_CASE(Algorithm::EltwiseErf, jit_erf_emitter),
        OV_CASE(Algorithm::EltwiseSoftSign, jit_soft_sign_emitter),
        OV_CASE(Algorithm::EltwiseIsFinite, jit_is_finite_emitter),
        OV_CASE(Algorithm::EltwiseIsInf, jit_is_inf_emitter),
        OV_CASE(Algorithm::EltwiseIsNaN, jit_is_nan_emitter),
        OV_CASE(Algorithm::EltwiseSelect, jit_select_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }

    inline void compute_eltwise_op() {
        std::vector<size_t> in_idxs;
        std::vector<size_t> aux_idxs;
        for (size_t i = 0; i < eltwise_emitter->get_inputs_num(); i++)
            in_idxs.push_back(get_vmm_reg(i).getIdx());
        for (size_t i = 0; i < eltwise_emitter->aux_vecs_count(); i++)
            aux_idxs.push_back(get_aux_vmm(i).getIdx());

        std::vector<size_t> out_idxs;
        out_idxs.push_back(vmm_dst.getIdx());

        eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs);
    }

    inline void apply_post_ops(bool is_scalar, int offset = 0) {
        int input_idx = eltwise_emitter->get_inputs_num();
        int eltwise_post_op_idx = 0;
        int quantization_post_op_idx = 0;
        for (size_t i = 1; i < ops_list_.size(); i++) {
            if (ops_list_[i] == ov::intel_cpu::Type::Eltwise) {
                std::vector<size_t> in_idxs;
                std::vector<size_t> aux_idxs;
                in_idxs.push_back(vmm_dst.getIdx());
                for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++)
                    in_idxs.push_back(get_vmm_reg(input_idx++).getIdx());
                for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_vecs_count(); j++)
                    aux_idxs.push_back(get_aux_vmm(j).getIdx());

                std::vector<size_t> out_idxs;
                out_idxs.push_back(vmm_dst.getIdx());

                post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_idxs);

                eltwise_post_op_idx++;
            } else if (ops_list_[i] == ov::intel_cpu::Type::FakeQuantize) {
                auto& p = post_ops_.get()->entry_[quantization_post_op_idx];
                bool do_dequantization = p.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || jep_.dst_prc == Precision::FP32 || i != ops_list_.size() - 1;
                int s_idx = vmm_dst.getIdx();

                size_t ptrs_table_off = quantization_post_op_idx * quantization_injectors[quantization_post_op_idx]->memoryStep();

                quantization_injectors[quantization_post_op_idx]->init_crop_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_crop(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_input_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_input_scale_shift(s_idx, s_idx + 1, offset, do_rounding,
                                                                                            is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_output_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_output_scale_shift(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_post_op_idx++;
            } else {
                IE_THROW(Unexpected) << "Eltwise jit kernel: unexpected operation type";
            }
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc, bool broadcast) {
        Xmm xmm_src = Xmm(vmm_src.getIdx());

        if (broadcast) {
            load_scalar(xmm_src, op, src_prc, dst_prc);
            uni_vbroadcastss(vmm_src, xmm_src);
        } else {
            switch (src_prc) {
                case Precision::FP32:
                case Precision::I32:
                    uni_vmovups(vmm_src, op);
                    break;
                case Precision::BF16:
                    vpmovzxwd(vmm_src, op);
                    uni_vpslld(vmm_src, vmm_src, 16);
                    break;
                case Precision::FP16:
                    vcvtph2ps(vmm_src, op);
                    break;
                case Precision::U16:
                    uni_vpmovzxwd(vmm_src, op);
                    break;
                case Precision::I16:
                    uni_vpmovsxwd(vmm_src, op);
                    break;
                case Precision::I8:
                    uni_vpmovsxbd(vmm_src, op);
                    break;
                case Precision::U8:
                    uni_vpmovzxbd(vmm_src, op);
                    break;
                default:
                    assert(!"unknown src_prc");
            }

            switch (dst_prc) {
                case Precision::FP32:
                    if (!src_prc.is_float())
                        uni_vcvtdq2ps(vmm_src, vmm_src);
                    break;
                case Precision::I32:
                    if (src_prc.is_float())
                        uni_vcvtps2dq(vmm_src, vmm_src);
                    break;
                default:
                    assert(!"unknown dst_prc");
            }
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(xmm_src, op);
                break;
            case Precision::BF16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case Precision::FP16:
                vcvtph2ps(xmm_src, op);
                break;
            case Precision::I16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovsxwd(xmm_src, op);
                break;
            case Precision::U16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovzxwd(xmm_src, op);
                break;
            case Precision::I8:
                movsx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            case Precision::U8:
                movzx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
                if (!src_prc.is_float())
                    uni_vcvtdq2ps(xmm_src, xmm_src);
                break;
            case Precision::I32:
                if (src_prc.is_float())
                    uni_vcvtps2dq(xmm_src, xmm_src);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, Precision src_prc, Precision dst_prc) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        switch (src_prc) {
            case Precision::FP32:
                if (!dst_prc.is_float())
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                break;
            case Precision::I32:
                if (dst_prc.is_float())
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, vmm_dst);
                break;
            case Precision::BF16:
                uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case Precision::FP16:
                vcvtps2ph(op, vmm_dst, 0x4);
                break;
            case Precision::I16:
                if (isa == x64::avx512_core) {
                    vpmovsdw(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::U16:
                if (isa == x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdw(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::I8:
                if (isa == x64::avx512_core) {
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case Precision::U8:
                if (isa == x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
                if (!dst_prc.is_float())
                    uni_vcvtps2dq(xmm_dst, xmm_dst);
                break;
            case Precision::I32:
                if (dst_prc.is_float())
                    uni_vcvtdq2ps(xmm_dst, xmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(op, xmm_dst);
                break;
            case Precision::BF16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                uni_vpextrw(op, xmm_dst, 0x0);
                break;
            case Precision::FP16:
                vcvtps2ph(xmm_dst, xmm_dst, 0x4);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_16);
                break;
            case Precision::I16:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_16);
                break;
            case Precision::U16:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_16);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }
};

}   // namespace intel_cpu
}   // namespace ov

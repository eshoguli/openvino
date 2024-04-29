// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

#include "emitters/plugin/riscv64/jit_add_emitter.hpp"
#include "emitters/plugin/riscv64/jit_divide_emitter.hpp"
#include "emitters/plugin/riscv64/jit_multiply_emitter.hpp"
#include "emitters/plugin/riscv64/jit_subtract_emitter.hpp"


namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

void jit_uni_eltwise_kernel::operator()(
    const node::jit_eltwise_call_args_ptrs* const_args,
    const jit_eltwise_call_args_indexes* indexes) {
    assert(ker_ && "jit_uni_eltwise_kernel: ker_ is null");
    for (auto i = 0; i < MAX_ELTWISE_DIM_RANK; ++i) {
        std::cout << indexes->indexes[i] << ", ";
    }
    std::cout << std::endl;

    //std::cout << "debug: jit_uni_eltwise_kernel::operator: start" << std::endl;
    ker_(const_args, indexes);
    //std::cout << "debug: jit_uni_eltwise_kernel::operator: end" << std::endl;
}

jit_uni_eltwise_generic::jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                                 const std::vector<EltwiseData>& eltwise_data,
                                                 const std::vector<ov::intel_cpu::Type>& ops_list,
                                                 const dnnl::post_ops& post_ops) :
                                                 jit_uni_eltwise_kernel(jep),
                                                 jit_generator(),
                                                 eltwise_data_(eltwise_data),
                                                 ops_list_(ops_list),
                                                 post_ops_(post_ops) {}

void jit_uni_eltwise_generic::generate() {
    std::cout << "debug: jit_uni_eltwise_generic::generate()" << std::endl;

    preamble();

    // addi(t1, x0, 4);
    // vsetvli(t0, t1, SEW::e32, LMUL::m8);

    // addi(sp, sp, -(4 * 32));
    // // vlw.v
    // vse32_v(vmm_dst, sp);

    // vle32_v(vmm_dst, sp);
    // addi(sp, sp, (4 * 32));


    //auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
    auto const exec_prc = ov::element::f32;

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    //eltwise_emitter = std::make_shared<ov::intel_cpu::riscv64::jit_add_emitter>(this, exec_prc);

    const auto &jep = jep_;

    //Reg param2 = abi_param2;
    Reg param1 = Xbyak_riscv::a0;
    Reg param2 = Xbyak_riscv::a1;
    const int offset_count = jep.input_size - 1;

    //vsetvli(t0, a3, SEW::e32, LMUL::m8);

    // ptrs initializing
    if (jep.use_runtime_ptrs) {
        assert(false && "unexpected");
    } else {
        auto init_ptrs_with_offsets = [this, offset_count, param2](Reg pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    Reg offset_reg(get_aux_gpr(0));
                    addi(offset_reg, x0, offsets[j]);

                    Reg index_reg(get_aux_gpr(1));
                    ld(index_reg, param2, static_cast<int32_t>(j * sizeof(size_t)));

                    mul(offset_reg, offset_reg, index_reg);
                    add(pointer, pointer, offset_reg);

                    std::cout << "init_ptrs_with_offsets: offset=" << (offsets[j] * j * sizeof(size_t)) << std::endl;
                }
            }
        };

        // TODO: debug only

        // // TODO: debug only
        // // value1
        // lw(s0, a0);
        // // value2
        // lw(s1, a0, 4);

        // // src0
        // const auto src_offset0 = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + 0 * sizeof(void*));
        // ld(a2, a0, src_offset0);
        // lw(s2, a2);
        // lw(s2, a2, 4);
        // lw(s2, a2, 8);

        for (size_t i = 0; i < jep.inputs_number; i++) {
            //ldr(get_src_reg(i), ptr(param1, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t))));
            //add(get_src_reg(i), x0, param1);
            const auto src_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t));
            std::cout << "debug: src_offset" << i << " = " << src_offset << std::endl;
            ld(get_src_reg(i), param1, src_offset);
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);

            // TODO: debug only
            // lw(s5, get_src_reg(i));
            // lw(s5, get_src_reg(i), 4);
            // lw(s5, get_src_reg(i), 8);
        }

        //ldr(reg_dst, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr))));
        //add(reg_dst, x0, param1);
        const auto dst_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr));
        std::cout << "debug: dst_offset = " << dst_offset << std::endl;
        ld(reg_dst, a0, dst_offset);
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        // //mov(reg_oc_off, 0);
        // init_ptrs_with_offsets(reg_oc_off, jep.oc_offsets);

        // //mov(reg_work_amount, jep.work_amount);
        std::cout << "jep.work_amount=" << jep.work_amount << std::endl;
        addi(t1, x0, jep.work_amount);

        // TODO: debug only
        // add(s5, x0, x0);
        // sw(s5, reg_dst);
    }

    Label unroll_loop_label;
    Label unroll_loop_end_label;
    Label main_loop_label;
    Label main_loop_end_label;
    Label tail_loop_label;
    Label tail_loop_end_label;

    const uint32_t vector_length_multiplier = 8;

    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            std::cout << "input " << i << ": scalar" << std::endl;
            vsetvli(t0, t1, SEW::e32, LMUL::m8);

            // TODO: workaround: broadcast load
            flw(f0, get_src_reg(i));
            const auto vmm_reg = get_vmm_reg(i * vector_length_multiplier);
            vxor_vv(vmm_reg, vmm_reg, vmm_reg);
            vfadd_vf(vmm_reg, vmm_reg, f0);

            // TODO: debug only
            // vse32_v(vmm_reg, reg_dst);
            // ld(reg_dst, a0, 0);
            // ld(reg_dst, a0, 1);
            // ld(reg_dst, a0, 2);
            // ld(reg_dst, a0, 3);
            // ld(reg_dst, a0, 4);
            // ld(reg_dst, a0, 5);
            // ld(reg_dst, a0, 6);
            // ld(reg_dst, a0, 7);
        }
    }

    size_t min_src_size = jep.dst_size;
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] != 1)
            min_src_size = std::min(min_src_size, jep.src_size[i]);
    }
    if (jep_.oc_size > 1)
        min_src_size = std::min(min_src_size, jep_.oc_size);

    if (min_src_size != jep.dst_size) {
        assert(false && "unexpected");
    }

    if (min_src_size == jep.dst_size) {
        // addi(t0, x0, 4 * 32); // not neccessary
        addi(t1, x0, jep.work_amount);

        L(main_loop_label);
        {
            vsetvli(t0, t1, SEW::e32, LMUL::m8);

            //const size_t vlen = cpu_isa_traits<isa>::vlen;
            //const size_t exec_prc_size = exec_prc.size();
            //const size_t loop_step = vlen / exec_prc_size;

            //cmp(reg_work_amount, loop_step);
            //b(LO, main_loop_end_label);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    std::cout << "input " << i << ": vector" << std::endl;
                    vle32_v(get_vmm_reg(i * vector_length_multiplier), get_src_reg(i));

                    lw(t6, get_src_reg(i), 0);
                    lw(t6, get_src_reg(i), 4);

                    add(get_src_reg(i), get_src_reg(i), t0);
                }
            }
            // vle32_v(v0, s2);
            // add(s2, s2, t0);

            // vle32_v(v8, s3);
            // add(s3, s3, t0);

            sub(t1, t1, t0);

            // TODO: here: uncomment
            compute_eltwise_op();

            //apply_post_ops();

            //store_vector(reg_dst, vmm_dst, exec_prc, jep.dst_prc);
            vse32_v(v0, reg_dst);

            // TODO: debug: v0 - OK
            // vse32_v(v8, reg_dst);
            // lw(t6, reg_dst);
            // lw(t6, reg_dst, 4);
            // vse32_v(v0, reg_dst);
            // lw(t6, reg_dst);
            // lw(t6, reg_dst, 4);


            // TODO: debug: v8 - OK
            // vse32_v(v0, reg_dst);
            // lw(t6, reg_dst);
            // lw(t6, reg_dst, 4);
            // vse32_v(v8, reg_dst);
            // lw(t6, reg_dst);
            // lw(t6, reg_dst, 4);


            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    //add(get_src_reg(i), get_src_reg(i), jep.src_prc[i].size() * loop_step);
                }
            }

            //add(reg_dst, reg_dst, jep.dst_prc.size() * loop_step);
            //sub(reg_work_amount, reg_work_amount, loop_step);
            if (jep_.oc_size > 1) {
                //add(reg_oc_off, reg_oc_off, loop_step * sizeof(float));
            }

            bnez(t1, main_loop_label);
        }
        L(main_loop_end_label);
    }

    postamble();

    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        post_op_emitters[i]->emit_data();
    }
}

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    ov::intel_cpu::riscv64::jit_generator *host;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.exec_prc);
    }
};

std::shared_ptr<jit_emitter> jit_uni_eltwise_generic::create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {
        nullptr,
        this,
        data,
        exec_prec
    };

    OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
    OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::riscv64::jit_add_emitter),
    OV_CASE(Algorithm::EltwiseDivide, ov::intel_cpu::riscv64::jit_divide_emitter),
    OV_CASE(Algorithm::EltwiseMultiply, ov::intel_cpu::riscv64::jit_multiply_emitter),
    OV_CASE(Algorithm::EltwiseSubtract, ov::intel_cpu::riscv64::jit_subtract_emitter));

    if (!ctx.emitter)
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");

    return ctx.emitter;
}

void jit_uni_eltwise_generic::compute_eltwise_op() {
    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_count(); i++) {
        in_idxs.push_back(get_vmm_reg(i).getIdx());
    }

    std::vector<size_t> aux_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_vecs_count(); i++) {
        aux_idxs.push_back(get_aux_vmm(i).getIdx());
    }

    std::vector<size_t> out_idxs;
    out_idxs.push_back(vmm_dst.getIdx());

    std::vector<size_t> gpr_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_gprs_count(); i++) {
        gpr_idxs.push_back(get_aux_gpr(i).getIdx());
    }

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs, gpr_idxs);
}

void jit_uni_eltwise_generic::load_vector(const VReg& data,
                                          const Reg& ptr_reg,
                                          const ov::element::Type& src_prc,
                                          const ov::element::Type& dst_prc,
                                          const bool broadcast,
                                          const int32_t ptr_offset) {
}

void jit_uni_eltwise_generic::store_vector(const Reg& ptr,
                                           const VReg& data,
                                           const ov::element::Type& src_prc,
                                           const ov::element::Type& dst_prc,
                                           const int32_t ptr_offset) {
}

namespace {
template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<std::vector<element::Type>> &precisions) {
        precisions = T::get_supported_precisions();
    }
};
}

ov::element::Type eltwise_precision_helper::get_precision(const size_t inputs_number,
                                                          const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                                          const std::vector<EltwiseData>& eltwise_data) {
    ov::element::Type exec_prc = ov::element::undefined;
    return exec_prc;
}

std::set<std::vector<element::Type>> eltwise_precision_helper::get_supported_precisions(const Algorithm& algo) {
    std::set<std::vector<element::Type>> precisions;

    OV_SWITCH(intel_cpu, SupportedPrecisions, precisions, algo,
              OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
              OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
              OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
              OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));
    if (precisions.empty())
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");

    return precisions;
}

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov

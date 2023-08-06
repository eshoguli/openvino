// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

// TODO remove
using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace InferenceEngine;

void jit_uni_eltwise_kernel::operator()(
    const node::jit_eltwise_call_args_ptrs* const_args,
    const jit_eltwise_call_args_indexes* indexes) {
    assert(ker_);

#ifdef DEBUG
    const auto src_ptr = static_cast<const float*>(const_args->src_ptr[0]);
    std::cout << "jit_uni_eltwise_kernel::operator(), src_ptr: " << std::endl;
    for (size_t i = 0; i < 8; i++) {
        std::cout << src_ptr[i] << " ";
    }
    std::cout << std::endl;
#endif // DEBUG

    ker_(const_args, indexes);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
jit_uni_eltwise_generic<isa>::jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                                      const std::vector<EltwiseData>& eltwise_data,
                                                      const std::vector<ov::intel_cpu::Type>& ops_list,
                                                      const dnnl::post_ops& post_ops) :
                                                      jit_uni_eltwise_kernel(jep),
                                                      jit_generator(),
                                                      eltwise_data_(eltwise_data),
                                                      ops_list_(ops_list),
                                                      post_ops_(post_ops) {
#ifdef DEBUG
    std::cout << "\tjit_uni_eltwise_generic:" << std::endl;
    for (const auto& eltwise_data_item : eltwise_data) {
        std::cout << "\t\talgo: " << algToString(eltwise_data_item.algo) << std::endl;
    }

    std::cout << "\t\tjep.work_amount:" << jep.work_amount << std::endl;
    std::cout << "\t\tjep.src_offsets:" << std::endl;
    for (const auto& src_offsets : jep.src_offsets) {
        if (src_offsets.empty()) {
            continue;
        }
        std::cout << "\t\t\t";
        for (const std::size_t offset : src_offsets) std::cout << offset << ",";
        std::cout << std::endl;
    }

    std::cout << "\t\tjep.dims: ";
    for (const auto& dim : jep.dims) std::cout << dim << ",";
    std::cout << std::endl;
#endif // DEBUG
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::generate() {
    const auto get_precision = []() {
        const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32;
        return exec_prc;
    };
    const auto exec_prc = get_precision();

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    }

    preamble();

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);

    XReg param2 = abi_param2;
    const int offset_count = jep_.input_size - 1;

    auto init_ptrs_with_offsets = [this, offset_count, param2](XReg pointer, const std::vector<size_t>& offsets) {
        for (int j = 0; j < offset_count; j++) {
            if (jep_.dims[j] != 1 && offsets[j] != 0) {
                XReg offset_reg(14);
                mov(offset_reg, offsets[j]); // x14 <= 16

                // TODO: do we really need X_TMP_0?
                add_imm(X_TMP_0, param2, j * sizeof(size_t), X_TMP_1); // x23 <= param2 + 16
                XReg index_reg(15);
                // x15 <= ptr(param2 + 16):
                // iter #1: index_reg = 0
                // iter #2: index_reg = 1
                ldr(index_reg, ptr(X_TMP_0));

                mul(offset_reg, offset_reg, index_reg); // x14 = param2
                add(pointer, pointer, offset_reg);

                // TODO: debug only
                //std::cout << offsets[j] << " x " << j * sizeof(size_t) << std::endl;
            }
        }
    };

    for (size_t i = 0; i < jep_.inputs_number; i++) {
        // TODO: do we really need X_TMP_0?
        add_imm(X_TMP_0, param1, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t), X_TMP_1);
        ldr(get_src_reg(i), ptr(X_TMP_0));

        init_ptrs_with_offsets(get_src_reg(i), jep_.src_offsets[i]);
    }

    add_imm(X_TMP_0, param1, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
    ldr(reg_dst, ptr(X_TMP_0));
    init_ptrs_with_offsets(reg_dst, jep_.dst_offsets);

    mov(reg_work_amount, jep_.work_amount);

    uni_ld1rw(get_vmm_reg(1).s4, get_src_reg(1), 0);

    Label main_loop_label;
    Label main_loop_end_label;
    L(main_loop_label);
    {
#ifdef DEBUG
        // marker
        mov(XReg(0), XReg(0));
        mov(XReg(0), XReg(0));
        mov(XReg(0), XReg(0));
#endif // DEBUG

        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t exec_prc_size = exec_prc.size();
        const size_t loop_step = vlen / exec_prc_size;

        cmp(reg_work_amount, loop_step);
        b(LO, main_loop_end_label);

        for (size_t i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_size[i] != 1) {
                uni_ldr(get_vmm_reg(i), get_src_reg(i), jep_.src_prc[i], exec_prc, false);
            }
        }

        compute_eltwise_op();

        apply_post_ops();

        uni_str(reg_dst, vmm_dst, exec_prc, jep_.dst_prc);

        for (size_t i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_size[i] != 1) {
                add(get_src_reg(i), get_src_reg(i), jep_.src_prc[i].size() * loop_step);
            }
        }

        add(reg_dst, reg_dst, jep_.dst_prc.size() * loop_step);

        sub(reg_work_amount, reg_work_amount, loop_step);

        b(AL, main_loop_label);

#ifdef DEBUG
        // marker
        mov(XReg(1), XReg(1));
        mov(XReg(1), XReg(1));
        mov(XReg(1), XReg(1));
#endif // DEBUG
    }
    L(main_loop_end_label);

    Label tail_loop_label;
    Label tail_loop_end_label;
    L(tail_loop_label);
    {
#ifdef DEBUG
        // marker
        mov(XReg(2), XReg(2));
        mov(XReg(2), XReg(2));
        mov(XReg(2), XReg(2));
#endif // DEBUG

        const size_t loop_step = 1;

        cmp(reg_work_amount, 0x0);
        b(EQ, tail_loop_end_label);

        // load scalar
        ////mov(get_vmm_reg(0).s, P_ALL_ONE / Xbyak_aarch64::T_z, 0x0);
        //ldr(get_scl_reg(0), ptr(get_src_reg(0)));
        for (size_t i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_size[i] != 1) {
                uni_ldr(get_scl_reg(i), get_src_reg(i), jep_.src_prc[i], exec_prc);
            }
        }

        //// ldr(get_vmm_reg(0).s, ptr(get_src_reg(0)));

        //// vmm_reg.d, mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src_reg)
        //// ld1rd(get_vmm_reg(0).s, mask / Xbyak_aarch64::T_z, get_src_reg(0));

        compute_eltwise_op();

        apply_post_ops();

        SReg sc_dst_reg{vmm_dst.getIdx()};
        uni_str(reg_dst, sc_dst_reg, exec_prc, jep_.dst_prc);

        for (size_t i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_size[i] != 1) {
                add(get_src_reg(i), get_src_reg(i), jep_.src_prc[i].size() * loop_step);
            }
        }

        add(reg_dst, reg_dst, jep_.dst_prc.size() * loop_step);

        // TODO: whilelo
        sub(reg_work_amount, reg_work_amount, loop_step);

        b(AL, tail_loop_label);

#ifdef DEBUG
        // marker
        mov(XReg(3), XReg(3));
        mov(XReg(3), XReg(3));
        mov(XReg(3), XReg(3));
#endif // DEBUG
    }
    L(tail_loop_end_label);

    postamble();
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::uni_ldr(const TReg& data, const XReg& ptr, const Precision& src_prc, const Precision& dst_prc, const bool broadcast) {
    if (broadcast) {
        IE_THROW(Unexpected) << "broadcast is not supported";
    }

    if (src_prc != dst_prc) {
        IE_THROW(Unexpected) << "src_prc != dst_prc is not supported";
    }

    switch (dst_prc) {
        case Precision::FP32: {
            jit_generator::uni_ldr(data, ptr);
            break;
        }
        default: {
            IE_THROW(Unexpected) << "dst_prc " << src_prc << " is not supported";;
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::uni_ldr(const SReg& data, const XReg& ptr, const Precision& src_prc, const Precision& dst_prc) {
    if (src_prc != dst_prc) {
        IE_THROW(Unexpected) << "src_prc != dst_prc is not supported";
    }

    switch (dst_prc) {
        case Precision::FP32: {
            ldr(data, Xbyak_aarch64::ptr(ptr));
            break;
        }
        default: {
            IE_THROW(Unexpected) << "dst_prc " << src_prc << " is not supported";;
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::uni_str(const XReg& ptr, const TReg& data, const Precision& src_prc, const Precision& dst_prc) {
    if (src_prc != dst_prc) {
        IE_THROW(Unexpected) << "src_prc != dst_prc is not supported";
    }

    switch (dst_prc) {
        case Precision::FP32: {
            jit_generator::uni_str(data, ptr);
            break;
        }
        default: {
            IE_THROW(Unexpected) << "dst_prc " << src_prc << " is not supported";;
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::uni_str(const XReg& ptr, const SReg& data, const Precision& src_prc, const Precision& dst_prc) {
    if (src_prc != dst_prc) {
        IE_THROW(Unexpected) << "uni_str: src_prc != dst_prc is not supported";
    }

    switch (dst_prc) {
        case Precision::FP32: {
            str(data, Xbyak_aarch64::ptr(ptr));
            break;
        }
        default: {
            IE_THROW(Unexpected) << "dst_prc " << src_prc << " is not supported";;
        }
    }
}

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    dnnl::impl::cpu::aarch64::jit_generator *host;
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa;
    const EltwiseData& opData;
    InferenceEngine::Precision exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
    }
};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
std::shared_ptr<jit_emitter> jit_uni_eltwise_generic<isa>::create_eltwise_emitter(const EltwiseData& data, const Precision& exec_prec) {
    EltwiseEmitterContext ctx = {
        nullptr,
        this,
        isa,
        data,
        exec_prec
    };

    OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
    OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::aarch64::jit_add_emitter),
    OV_CASE(Algorithm::EltwiseMulAdd, ov::intel_cpu::aarch64::jit_mul_add_emitter),
    OV_CASE(Algorithm::EltwiseMultiply, ov::intel_cpu::aarch64::jit_multiply_emitter));

    if (!ctx.emitter)
        IE_THROW() << "Unsupported operation type '" << algToString(data.algo) << "' for Eltwise emitter";

    return ctx.emitter;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::compute_eltwise_op() {
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::apply_post_ops() {
    int input_idx = eltwise_emitter->get_inputs_num();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < ops_list_.size(); i++) {
        // TODO: FakeQuantize is not supported
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
        } else {
            IE_THROW(Unexpected) << "Eltwise jit kernel: unexpected operation type";
        }
    }
}

template struct jit_uni_eltwise_generic<cpu_isa_t::asimd>;
template struct jit_uni_eltwise_generic<cpu_isa_t::sve_128>;
template struct jit_uni_eltwise_generic<cpu_isa_t::sve_256>;
// TODO: oneDNN doesn't support
//template struct jit_uni_eltwise_generic<cpu_isa_t::sve_384>;
template struct jit_uni_eltwise_generic<cpu_isa_t::sve_512>;

}  // namespace aarch64
}  // namespace intel_cpu
}  // namespace ov

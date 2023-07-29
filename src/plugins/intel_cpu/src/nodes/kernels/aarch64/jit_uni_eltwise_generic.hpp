// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>

#include <emitters/aarch64/jit_emitter.hpp>
#include <emitters/aarch64/jit_eltwise_emitters.hpp>

#include <onednn/dnnl.h>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include "nodes/kernels/jit_eltwise_call_args_ptrs.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

// TODO: not completed
#define MAX_ELTWISE_INPUTS 7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
    bool use_runtime_ptrs;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

struct jit_uni_eltwise_kernel {
    void (*ker_)(const node::jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const node::jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);

        // TODO: debug only
        const auto src_ptr = static_cast<const float*>(const_args->src_ptr[0]);
        std::cout << std::endl;
        for (size_t i = 0; i < 16; i++) {
            std::cout << src_ptr[i] << " ";
        }
        std::cout << std::endl;

        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

struct EltwiseData {
    Algorithm algo;
    dnnl::algorithm onednnAlgorithm;
    float alpha;
    float beta;
    float gamma;

    bool operator==(const EltwiseData& rhs) const noexcept {
        return algo == rhs.algo &&
            onednnAlgorithm == rhs.onednnAlgorithm &&
            alpha == rhs.alpha &&
            beta == rhs.beta &&
            gamma == rhs.gamma;
    }
};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                     const std::vector<EltwiseData>& eltwise_data,
                                     const std::vector<ov::intel_cpu::Type>& ops_list,
                                     const dnnl::post_ops& post_ops) :
                                     jit_uni_eltwise_kernel(jep),
                                     jit_generator(),
                                     eltwise_data_(eltwise_data),
                                     ops_list_(ops_list),
                                     post_ops_(post_ops) {}

    jit_uni_eltwise_generic() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    // void generate() override {
    //     // TODO: not implemented

    //     // auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
    //     auto const exec_prc = InferenceEngine::Precision::FP32;

    //     eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    //     // for (size_t i = 1; i < eltwise_data_.size(); ++i) {
    //     //     post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    //     // }

    //     // jit_generator::preamble
    //     preamble();

    //     XReg param = param1;
    //     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr), X_TMP_1);
    //     ldr(reg_src, ptr(X_TMP_0));

    //     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
    //     ldr(reg_dst, ptr(X_TMP_0));

    //     ldr(vmm_src, ptr(reg_src));

    //     str(vmm_src, ptr(reg_dst));

    //     compute_eltwise_op();

    //     // jit_generator::postamble
    //     postamble();
    // }

    //void generate() override;

    // void generate() override {
    //     // TODO: not implemented

    //     // auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
    //     auto const exec_prc = InferenceEngine::Precision::FP32;

    //     eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    //     // for (size_t i = 1; i < eltwise_data_.size(); ++i) {
    //     //     post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    //     // }

    //     // jit_generator::preamble
    //     preamble();

    //     mov(x0, x0);
    //     mov(x0, x0);
    //     mov(x0, x0);

    //     XReg param = param1;
    //     // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr), X_TMP_1);
    //     // ldr(reg_src, ptr(X_TMP_0));

    //     // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + sizeof(size_t), X_TMP_1);
    //     // ldr(reg_src1, ptr(X_TMP_0));

    //     for (size_t i = 0; i < jep_.inputs_number; i++) {
    //         add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t), X_TMP_1);
    //         ldr(get_src_reg(i), ptr(X_TMP_0));
    //     }

    //     // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[1]), X_TMP_1);
    //     // ldr(reg_src1, ptr(X_TMP_0));

    //     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
    //     ldr(reg_dst, ptr(X_TMP_0));

    //     mov(x3, jep_.work_amount);

    //     ldr(vmm_src, ptr(reg_src));
    //     ldr(x_src1, ptr(reg_src1));

    //     mov(x0, x0);
    //     mov(x0, x0);
    //     mov(x0, x0);

    //     str(vmm_src, ptr(reg_dst));

    //     compute_eltwise_op();
    //     mov(x0, x0);
    //     mov(x0, x0);
    //     mov(x0, x0);

    //     // jit_generator::postamble
    //     postamble();
    // }

    void generate() override {
        // TODO: not implemented

        const auto get_precision = []() {
            const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32;
            return exec_prc;
        };

        const auto exec_prc = get_precision();
        //const auto exec_prc = InferenceEngine::Precision::FP32;

        eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
        // for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        //     post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
        // }

        // jit_generator::preamble
        preamble();

        mov(x0, x0);
        mov(x0, x0);
        mov(x0, x0);

        XReg param = param1;
        // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr), X_TMP_1);
        // ldr(reg_src, ptr(X_TMP_0));

        // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + sizeof(size_t), X_TMP_1);
        // ldr(reg_src1, ptr(X_TMP_0));

        for (size_t i = 0; i < jep_.inputs_number; i++) {
            add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t), X_TMP_1);
            ldr(get_src_reg(i), ptr(X_TMP_0));

            // TODO: explore
            //tst(X_TMP_1, X_TMP_1);
        }

        // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[1]), X_TMP_1);
        // ldr(reg_src1, ptr(X_TMP_0));

        add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
        ldr(reg_dst, ptr(X_TMP_0));

        mov(reg_work_amount, jep_.work_amount);

        mov(x0, x0);
        mov(x0, x0);
        mov(x0, x0);

        Label main_loop_label;
        Label main_loop_end_label;
        L(main_loop_label);
        {
            const size_t vlen = cpu_isa_traits<isa>::vlen;
            const size_t exec_prc_size = exec_prc.size();
            const size_t loop_step = vlen / exec_prc_size;

            cmp(reg_work_amount, 0x0);
            //tst(reg_work_amount, loop_step);
            b(EQ, main_loop_end_label);

            ldr(vmm_src, ptr(reg_src));
            ldr(x_src1, ptr(reg_src1));

            str(vmm_src, ptr(reg_dst));


            // TODO: just to test
            const auto offset = jep_.dst_prc.size() * loop_step;
            add(reg_dst, reg_dst, offset);

            sub(reg_work_amount, reg_work_amount, loop_step);

            b(AL, main_loop_label);
        }
        L(main_loop_end_label);

        compute_eltwise_op();
        mov(x0, x0);
        mov(x0, x0);
        mov(x0, x0);

        // jit_generator::postamble
        postamble();
    }

private:
    //using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    using TReg = QReg;
    // using TRegS = typename cpu_isa_traits<isa>::TRegS;

    Xbyak_aarch64::XReg reg_src = x11;
    Xbyak_aarch64::XReg reg_src1 = x12;
    //const XReg reg_src = x8;
    //const XReg reg_src1 = x9;
    const XReg reg_work_amount = x7;
    Xbyak_aarch64::XReg reg_dst = x8;

    Xbyak_aarch64::VReg4S xmm_src {1};
    TReg vmm_src {1};
    TReg vmm_src0 {1};
    TReg vmm_src1 {2};
    XReg x_src1 {9};

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

    XReg get_src_reg(int idx) {
        return XReg(reg_src.getIdx() + idx);
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, InferenceEngine::Precision exec_prec) {
        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            data,
            exec_prec
        };

        OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
        OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::aarch64::jit_add_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }

    inline void compute_eltwise_op() {
        // TODO: not completed

        std::vector<size_t> in_idxs;
        std::vector<size_t> aux_idxs;
        for (size_t i = 0; i < eltwise_emitter->get_inputs_num(); i++)
            in_idxs.push_back(i);
        for (size_t i = 0; i < eltwise_emitter->aux_vecs_count(); i++)
            aux_idxs.push_back(i);

        std::vector<size_t> out_idxs;
        out_idxs.push_back(0);

        eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs);
    }

    const std::vector<EltwiseData>& eltwise_data_;
    const std::vector<ov::intel_cpu::Type>& ops_list_;
    const dnnl::post_ops& post_ops_;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

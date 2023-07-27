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
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, dnnl::impl::cpu::aarch64::jit_generator {
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

    void generate() override {
        // TODO: not implemented

        // auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
        auto const exec_prc = InferenceEngine::Precision::FP32;

        eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
        // for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        //     post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
        // }

        // jit_generator::preamble
        preamble();


        // mov(x0, x0);
        // XReg param = param1;
        // add_imm(X_TMP_0, param, GET_OFF(src), X_TMP_1);
        // ldr(reg_src, ptr(X_TMP_0));
        // add_imm(X_TMP_0, param, GET_OFF(dst), X_TMP_1);
        // ldr(reg_dst, ptr(X_TMP_0));

        compute_eltwise_op();

        // jit_generator::postamble
        postamble();
    }

private:
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

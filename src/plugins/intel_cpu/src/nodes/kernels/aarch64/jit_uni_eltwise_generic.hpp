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

// TODO: x64 is used

#include <onednn/dnnl.h>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "emitters/x64/jit_emitter.hpp"
#include "emitters/x64/jit_eltwise_emitters.hpp"
#include "emitters/x64/jit_dnnl_emitters.hpp"

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

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

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
    //ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;

    // shape agnostic kernel
    size_t work_amount;
    const void *src_offsets[MAX_ELTWISE_INPUTS];
    const void *dst_offsets;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

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

struct EltwiseData {};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_eltwise_generic {};

// template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
// struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, public jit_generator {
//     DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

//     explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
//                                      const std::vector<EltwiseData>& eltwise_data,
//                                      const std::vector<ov::intel_cpu::Type>& ops_list,
//                                      const dnnl::post_ops& post_ops)
//     : jit_uni_eltwise_kernel(jep), jit_generator(jit_name()), eltwise_data_(eltwise_data), ops_list_(ops_list), post_ops_(post_ops) {}

//     void create_ker() override {
//         jit_generator::create_kernel();
//         ker_ = (decltype(ker_))jit_ker();
//     }

//     void generate() override {}
// };

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

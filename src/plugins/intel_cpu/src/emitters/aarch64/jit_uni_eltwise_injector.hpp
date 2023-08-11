// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    jit_uni_eltwise_injector_f32(jit_generator *host, 
                                 alg_kind_t alg,
                                 float alpha,
                                 float beta)
        : alg(alg), alpha(alpha), beta(beta), scale(scale), h(host) {
        // assert(eltwise_injector::is_supported(isa, alg_));
        // register_table_entries();
    }

private:
    const alg_kind_t alg;
    const float alpha;
    const float beta;
    const float scale;

    void exp_compute_vector_fwd(const TRegS &vmm_src);
    void tanh_compute_vector_fwd(const TRegS &vmm_src);
};

} // namespace aarch64
} // namespace intel_cpu
} // namespace ov

#endif

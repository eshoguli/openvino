// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_injector.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(const TRegS &vmm_src) {
    // const auto &t0 = TRegS(IDX(vmm_src));
    // const auto &t1 = TRegS(IDX(vmm_aux1));
    // const auto &t2 = TRegS(IDX(vmm_aux2));
    // h->fmin(t0, p_all, TRegS(IDX(table_val(exp_ln_flt_max_f, z_tmp))));
    // h->fmax(t0, p_all, TRegS(IDX(table_val(exp_ln_flt_min_f, z_tmp))));
    // h->fmul(t0, t0, TRegS(IDX(table_val(exp_log2ef, z_tmp))));
    // h->movprfx(t1, p_all, t0);
    // h->frintm(t1, p_all, t0);
    // h->fcvtzs(t2, p_all, t1);
    // h->fsub(t1, t0, t1);
    // h->fadd(t0, t1, TRegS(IDX(table_val(one, z_tmp))));
    // h->lsr(t1, t0, 17);
    // h->fexpa(t1, t1);
    // h->fscale(t1, p_all, t2);
    // h->and_(TRegD(t2.getIdx()), ZRegD(t0.getIdx()), TRegD(IDX(table_val(exp_not_mask17, z_tmp))));
    // h->fsub(t2, t0, t2);
    // h->movprfx(t0, p_all, TRegS(IDX(table_val(exp_coeff2, z_tmp))));
    // h->fmad(t0, p_all, t2, TRegS(IDX(table_val(exp_coeff1, z_tmp))));
    // h->fmad(t0, p_all, t2, TRegS(IDX(table_val(one, z_tmp))));
    // h->fmul(t0, t1, t0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(const TRegS &vmm_src) {
    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    // const auto &t0 = TRegS(IDX(vmm_src));
    // const auto &t1 = TRegS(IDX(vmm_aux1));
    // const auto &t2 = TRegS(IDX(vmm_aux2));
    // const auto &t3 = TRegS(IDX(vmm_aux3));
    // const auto &oneS = TRegS(IDX(vmm_aux4));
    // const auto &mask = PReg(6); // avoid pred regs used in *conv_kernel*

    // h->fcpy(oneS, p_all, 1);
    // // make mask for small x
    // h->mov(t3, p_all, t0);
    // h->fabs(t1, p_all, t0);
    // h->cmplt(mask.s, p_all, t1, TRegS(IDX(table_val(tanh_range, z_tmp))));

    // // 2x
    // h->fadd(t0, t0, t0);
    // // exp(2x)
    // exp_compute_vector_fwd(t0);
    // // 1+exp(2x)
    // h->fadd(t0, t0, oneS);
    // // 1/(1+exp(2x))
    // // 1st aprox ; a = 1/x + e
    // h->frecpe(t1, t0);
    // // 2nd aprox ; a' = (2 - ax)a = 1/x - e^2 x
    // h->frecps(t2, t0, t1);
    // h->fmul(t2, t2, t1);
    // // 3rd aprox ; a'' = (2 - a'x)a'
    // h->frecps(t0, t0, t2);
    // h->fmul(t0, t0, t2);

    // // 2/(1+exp(2x))
    // h->fadd(t0, t0, t0);
    // // 1-2/(1+exp(2x))
    // h->fsub(t0, oneS, t0);

    // // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    // h->fmul(t1, t3, t3);
    // h->fmad(t1, p_all, TRegS(IDX(table_val(tanh_m1d3, z_tmp))), oneS);
    // h->fmul(t1, p_all, t3);
    // // select the correct value according to mask
    // h->mov(t0, mask, t1);
}

template struct jit_uni_eltwise_injector_f32<asimd>;

} // namespace aarch64
} // namespace intel_cpu
} // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_injector.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

#define IDX(a) static_cast<uint32_t>(a.getIdx())

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(const TReg& src,
                                                               const TReg& dst,
                                                               const TReg& aux1,
                                                               const TReg& aux2) {
    // const auto &t0 = TReg(IDX(src));
    // const auto &t1 = TReg(IDX(aux1));
    // const auto &t2 = TReg(IDX(aux2));

    // h->fmin(src, p_all, TReg(IDX(table_val(exp_ln_flt_max_f, z_tmp))));
    //h->fmax(src, p_all, TReg(IDX(table_val(exp_ln_flt_min_f, z_tmp))));

    table_val(exp_ln_flt_min_f, aux1);
    h->fminnm(src.s, src.s, aux1.s);
    h->fmaxnm(src.s, src.s, aux1.s);

    // h->fmul(src, src, TRegS(IDX(table_val(exp_log2ef, z_tmp))));
    // h->movprfx(aux1, p_all, src);
    // h->frintm(aux1, p_all, src);
    // h->fcvtzs(aux2, p_all, aux1);
    // h->fsub(aux1, src, aux1);
    // h->fadd(src, aux1, TRegS(IDX(table_val(one, z_tmp))));
    // h->lsr(aux1, src, 17);
    // h->fexpa(aux1, aux1);
    // h->fscale(aux1, p_all, aux2);
    // h->and_(TRegD(aux2.getIdx()), ZRegD(src.getIdx()), TRegD(IDX(table_val(exp_not_mask17, z_tmp))));
    // h->fsub(aux2, src, aux2);
    // h->movprfx(src, p_all, TRegS(IDX(table_val(exp_coeff2, z_tmp))));
    // h->fmad(src, p_all, aux2, TRegS(IDX(table_val(exp_coeff1, z_tmp))));
    // h->fmad(src, p_all, aux2, TRegS(IDX(table_val(one, z_tmp))));
    // h->fmul(src, aux1, src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(const TReg& vmm_src, const TReg& vmm_dst) {
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

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };
    // exp(x) constants2
    static const table_t exp_consts2 {
            {exp_coeff1, {0x3f31721c, true}},
            {exp_coeff2, {0x3e772df2, true}},
            {exp_not_mask17, {~((1u << 17) - 1), true}},
    };
}

template struct jit_uni_eltwise_injector_f32<asimd>;

} // namespace aarch64
} // namespace intel_cpu
} // namespace ov

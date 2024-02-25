// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_injectors.hpp"

#include <memory>
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

size_t jit_exp_injector::get_aux_vecs_count() { return 4; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_exp_injector::emit_impl(dnnl::impl::cpu::aarch64::jit_generator* h,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
                                 const ov::element::Type exec_prc,
                                 const std::vector<size_t> &in_vec_idxs,
                                 const std::vector<size_t> &aux_vec_idxs,
                                 const std::vector<size_t> &out_vec_idxs,
                                 const Xbyak_aarch64::XReg& p_table) {
    const auto table_val2 = [&](const std::string& key, const size_t key_off_val_shift = 0) -> Xbyak_aarch64::AdrNoOfs {
        const auto it = entry_map.find(key); // search an entry for a key
        assert(it != entry_map.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? 16 : sizeof(jit_emitter::table_entry_val_t);
        const int32_t off = te.off + key_off_val_shift * scale;

        h->add_imm(h->X_DEFAULT_ADDR, p_table, off, h->X_TMP_0);
        return Xbyak_aarch64::ptr(h->X_DEFAULT_ADDR);
    };

    if (host_isa != dnnl::impl::cpu::aarch64::asimd) {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }

    if (exec_prc != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc.to_string());
    }

    //using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);

    const TReg vmm_mask(aux_vec_idxs[3]);

    h->ld1r(vmm_aux0.s, table_val2("exp_ln_flt_max_f"));
    h->fmin(vmm_dst.s, vmm_src.s, vmm_aux0.s);
    h->ld1r(vmm_aux0.s, table_val2("exp_ln_flt_min_f"));

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->facgt(vmm_mask.s, vmm_aux0.s, vmm_src.s);

    h->fmax(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
    h->mov(vmm_aux1.b16, vmm_dst.b16);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->ld1r(vmm_aux0.s, table_val2("exp_log2ef"));
    h->ld1r(vmm_aux2.s, table_val2("half"));
    h->fmla(vmm_aux2.s, vmm_dst.s, vmm_aux0.s);

    // tmp = floorf(fx)
    h->frintm(vmm_aux2.s, vmm_aux2.s);

    // keep vmm_src = fx for further computations
    h->mov(vmm_dst.b16, vmm_aux2.b16);

    // x = x - fx * ln2
    h->ld1r(vmm_aux0.s, table_val2("ln2f"));
    h->fmls(vmm_aux1.s, vmm_aux2.s, vmm_aux0.s);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->ld1r(vmm_aux0.s, table_val2("one"));
    h->fsub(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
    h->fcvtzs(vmm_aux2.s, vmm_dst.s);

    h->ld1r(vmm_aux0.s, table_val2("exponent_bias"));
    h->add(vmm_aux2.s, vmm_aux2.s, vmm_aux0.s);

    h->sqshl(vmm_aux2.s, vmm_aux2.s, 23);

    // set zeroes at those points which were < log(FLT_MIN)
    h->and_(vmm_aux2.b16, vmm_mask.b16, vmm_aux2.b16);

    // compute polynomial
    h->ld1r(vmm_aux0.s, table_val2("exp_pol5"));
    h->ld1r(vmm_dst.s, table_val2("exp_pol4"));
    h->fmla(vmm_dst.s, vmm_aux1.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table_val2("exp_pol3"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table_val2("exp_pol2"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    h->ld1r(vmm_aux0.s, table_val2("exp_pol1"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table_val2("one"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    // y = y * 2^n
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux2.s);
    h->ld1r(vmm_aux0.s, table_val2("two"));
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
}

void jit_exp_injector::push_entry_map(std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map) {
    const auto push_arg_entry_of = [&](const std::string key, const jit_emitter::table_entry_val_t val, const bool broadcast) {
        jit_emitter::mapped_table_entry_t te {0, val, broadcast};
        entry_map.insert(std::make_pair(key, te));
    };

    push_arg_entry_of("exp_ln_flt_max_f", 0x42b17218, true);
    push_arg_entry_of("exp_ln_flt_min_f", 0xc2aeac50, true);
    push_arg_entry_of("exp_log2ef", 0x3fb8aa3b, true);
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("two", 0x40000000, true);
    push_arg_entry_of("half", 0x3f000000, true);
    push_arg_entry_of("ln2f", 0x3f317218, true);
    push_arg_entry_of("exponent_bias", 0x0000007f, true);
    push_arg_entry_of("exp_pol1", 0x3f7ffffb, true);
    push_arg_entry_of("exp_pol2", 0x3efffee3, true);
    push_arg_entry_of("exp_pol3", 0x3e2aad40, true);
    push_arg_entry_of("exp_pol4", 0x3d2b9d0d, true);
    push_arg_entry_of("exp_pol5", 0x3c07cfce, true);
}

template void jit_exp_injector::emit_impl<dnnl::impl::cpu::aarch64::asimd>(
    dnnl::impl::cpu::aarch64::jit_generator* h,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
    const ov::element::Type exec_prc,
    const std::vector<size_t> &in_vec_idxs,
    const std::vector<size_t> &aux_vec_idxs,
    const std::vector<size_t> &out_vec_idxs,
    const Xbyak_aarch64::XReg& p_table);

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

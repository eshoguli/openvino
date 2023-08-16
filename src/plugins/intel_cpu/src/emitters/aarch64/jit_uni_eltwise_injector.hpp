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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    using TRegS = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TRegS;

    jit_uni_eltwise_injector_f32(dnnl::impl::cpu::aarch64::jit_generator *host,
                                 dnnl::impl::alg_kind_t alg,
                                 float alpha,
                                 float beta,
                                 Xbyak_aarch64::XReg x_table = Xbyak_aarch64::XReg(0))
        : h(host), alg(alg), alpha(alpha), beta(beta), x_table(x_table) {
        // assert(eltwise_injector::is_supported(isa, alg_));
        // register_table_entries();
    }

    // TODO: debug: temporary call directly
    void exp_compute_vector_fwd(const TReg& src, const TReg& dst, const TReg& aux1, const TReg& aux2);
    void tanh_compute_vector_fwd(const TReg& src, const TReg& dst);

    void register_table_entries();

private:
    dnnl::impl::cpu::aarch64::jit_generator* h;
    const dnnl::impl::alg_kind_t alg;
    const float alpha;
    const float beta;

    // TODO: not completed
    const Xbyak_aarch64::XReg x_table;
    const size_t vlen = dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::vlen;

    enum key_t {
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        exp_coeff1, // 0.6931473921 (0x3f31721c)
        exp_coeff2, // 0.2413862043 (0x3e772df2)
        exp_not_mask17, // ~((1u << 17) - 1)
    };

    size_t table_off(key_t key, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }

    // TODO: changed to TReg
    TReg table_val(key_t key, TReg zreg, size_t key_off_val_shift = 0) {
        Xbyak_aarch64::XReg x_addr(h->X_DEFAULT_ADDR);
        auto off = table_off(key, key_off_val_shift);

        if (off) {
            h->add_imm(x_addr, x_table, off, h->X_TMP_0);
        } else {
            x_addr = x_table;
        }

        // h->ldr(TReg(zreg.getIdx()), ptr(x_addr));
        // TODO: Load one single-element structure and Replicate to all lanes (of one register)
        h->ldr(Xbyak_aarch64::QReg(zreg.getIdx()), Xbyak_aarch64::ptr(x_addr));
        return zreg;
    }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<key_t, table_entry_t>;
    using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;

    mapped_table_t entry_map_;
};

} // namespace aarch64
} // namespace intel_cpu
} // namespace ov

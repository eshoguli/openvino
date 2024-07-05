// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include "xbyak_riscv/xbyak_riscv.hpp"
#include "../src/common/c_types_map.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

enum cpu_isa_bit_t : unsigned {
    asimd_bit = 1u << 0,
    sve_128_bit = 1u << 1,
    sve_256_bit = 1u << 2,
    sve_384_bit = 1u << 3,
    sve_512_bit = 1u << 4,
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    asimd = asimd_bit,
    sve_128 = sve_128_bit | asimd,
    sve_256 = sve_256_bit | sve_128,
    sve_384 = sve_384_bit | sve_256,
    sve_512 = sve_512_bit | sve_384,
    isa_all = ~0u,
};

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(gen_name) \
    const char *name() const override { return STRINGIFY(gen_name); } \
    const char *source_file() const override { return __FILE__; }     \
    static const char *jit_name() { \
        static constexpr char ret[] = "/oneDNN:" STRINGIFY(gen_name); \
        return ret; \
    }

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_default;
    static constexpr const char *user_option_env = "default";
};

template <>
struct cpu_isa_traits<asimd> {
    typedef Xbyak_riscv::VReg TReg;
//    typedef Xbyak_aarch64::VReg16B TRegB;
//    typedef Xbyak_aarch64::VReg8H TRegH;
//    typedef Xbyak_aarch64::VReg4S TRegS;
//    typedef Xbyak_aarch64::VReg2D TRegD;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    //static constexpr dnnl_cpu_isa_t user_option_val = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_asimd);
    static constexpr const char *user_option_env = "advanced_simd";
};

class jit_generator : public Xbyak_riscv::CodeGenerator {
public:
    const uint8_t *jit_ker() const {
        assert(jit_ker_ && "jit_ker_ is nullable");
        return jit_ker_;
    }

    void preamble();
    void postamble();

    void L(const char *label) = delete;
    void L(Xbyak_riscv::Label &label) {
        Xbyak_riscv::CodeGenerator::L(label);
    }

    virtual void create_kernel();

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;

    static inline bool is_initialized() {
        /* At the moment, Xbyak_aarch64 does not have GetError()\
         so that return dummy result. */
        return true;
    }

private:
    const uint8_t* getCode();
};

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov

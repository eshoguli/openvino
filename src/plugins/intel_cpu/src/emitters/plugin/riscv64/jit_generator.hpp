// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include "xbyak_riscv/xbyak_riscv.hpp"
// #include "oneapi/dnnl/dnnl_common_types.h"
#include "../src/common/c_types_map.hpp"
//#include "../src/cpu/jit_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

namespace {
// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
//static const Reg abi_param1(a0.getIdx());
//static const Reg abi_param2(a1.getIdx());
//static const Reg abi_param3(a2.getIdx());
//        abi_param4(Xbyak_riscv::a3),
//        abi_param5(Xbyak_riscv::a4),
//        abi_param6(Xbyak_riscv::a5),
//        abi_param7(Xbyak_riscv::a6),
//        abi_param8(Xbyak_riscv::a7);
} // namespace

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

    //virtual dnn::impl::status_t create_kernel();
    virtual void create_kernel();

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

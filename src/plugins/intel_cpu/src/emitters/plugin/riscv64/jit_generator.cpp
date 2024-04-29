// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_generator.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// src/plugins/intel_cpu/thirdparty/onednn/src/common/c_types_map.hpp

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

static const std::vector<Reg> store_gpr_regs = {
    // Temporaries. Saver: Caller
    t0, t1, t2, t3, t4, t5, t6,
    // Function arguments. Saver: Caller
    a0, a1, a2, a3, a4, a5, a6, a7,
    // Saved registers. Saver: Callee
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11
};

static const std::vector<VReg> vec_regs = {};

void jit_generator::preamble() {
    //addi(sp, sp, -8);
    //sw(t0, sp, 0);
    //sw(t1, sp, 4);

    addi(sp, sp, -store_gpr_regs.size() * 4);

    int imm = 0;
    for (const auto& reg : store_gpr_regs) {
        sw(reg, sp, imm);
        imm += 4;
    }
}

void jit_generator::postamble() {
    int imm = 0;
    for (const auto& reg : store_gpr_regs) {
        lw(reg, sp, imm);
        imm += 4;
    }

    addi(sp, sp, store_gpr_regs.size() * 4);

    //lw(t0, sp, 0);
    //lw(t1, sp, 4);
    //addi(sp, sp, 8);

    ret();
}

// dnn::impl::status_t jit_generator::create_kernel() {
//     generate();
//     jit_ker_ = getCode();
//     return (jit_ker_) ? dnn::impl::status::success : dnn::impl::status::runtime_error;
// }

void jit_generator::create_kernel() {
    generate();
    jit_ker_ = getCode();
    assert(jit_ker_ && "jit kernel is empty");
    //return (jit_ker_) ? dnn::impl::status::success : dnn::impl::status::runtime_error;
}

const uint8_t* jit_generator::getCode() {
    //std::cout << "debug: jit_generator::getCode()" << std::endl;

    this->ready();
    if (!is_initialized()) return nullptr;
    const uint8_t *code = reinterpret_cast<const uint8_t *>(CodeGenerator::getCode());
    //register_jit_code(code, getSize() * CSIZE);
    return code;
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov

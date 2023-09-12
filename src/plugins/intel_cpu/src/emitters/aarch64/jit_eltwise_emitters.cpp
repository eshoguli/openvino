// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <memory>
#include "ie_ngraph_utils.hpp"
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

namespace {
InferenceEngine::Precision get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<InferenceEngine::Precision> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(
            InferenceEngine::details::convertPrecision(input.get_source_output().get_element_type()));
    }

    assert(std::all_of(
        input_precisions.begin(),
        input_precisions.end(),
        [&input_precisions](const InferenceEngine::Precision& precision) {return precision == input_precisions[0]; }));

    return input_precisions[0];
}
} // namespace

/// ADD ///
jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node,
                                 const float alpha)
                                 : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node), alpha) {
}

jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const Precision exec_prc,
                                 const float alpha) : jit_emitter(host, host_isa, exec_prc, alpha) {
}

size_t jit_add_emitter::get_inputs_count() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fadd(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}};
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node,
                                         const float alpha)
                                         : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node), alpha) {
}

jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const Precision exec_prc,
                                         const float alpha)
                                         : jit_emitter(host, host_isa, exec_prc, alpha) {
}

size_t jit_mul_add_emitter::get_inputs_count() const { return 3; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg src2 = TReg(in_vec_idxs[2]);
    TReg dst = TReg(out_vec_idxs[0]);

    // uni_fmad implementation
    h->fmul(dst.s, src0.s, src1.s);
    h->fadd(dst.s, dst.s, src2.s);
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node,
                                           const float alpha)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node), alpha) {}

jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const Precision exec_prc,
                                           const float alpha)
                                           : jit_emitter(host, host_isa, exec_prc, alpha) {}

size_t jit_multiply_emitter::get_inputs_count() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fmul(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}};
}

/// POWER ///
jit_power_emitter::jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float power,
                                     const std::shared_ptr<ov::Node>& node)
                                     : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)), power(power) {
}

jit_power_emitter::jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float power,
                                     const Precision exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc), power(power) {
}

size_t jit_power_emitter::get_inputs_count() const { return 1; }

size_t jit_power_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_power_emitter::get_aux_gprs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_power_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_power_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}


extern "C" float my_function(float v1, float v2);
float my_function(float v1, float v2) {
    return pow(v1, v2);
}

namespace {
// void preamble(dnnl::impl::cpu::aarch64::jit_generator* h) {
//     using namespace Xbyak_aarch64::util;
//     uint64_t sveLen = get_sve_length();

//     h->stp(h->x29, h->x30, pre_ptr(h->sp, -16));
//     /* x29 is a frame pointer. */
//     h->mov(h->x29, h->sp);
//     h->sub(h->sp, h->sp, static_cast<int64_t>(preserved_stack_size) - 16);

//     /* x9 can be used as a temporal register. */
//     h->mov(h->x9, h->sp);

//     // if (vreg_to_preserve) {
//     //     st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
//     //     st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
//     // }
//     for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
//         stp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
//                 Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
//                 post_ptr(x9, xreg_len * 2));
//     }

//     // if (sveLen) { /* SVE is available. */
//     //     ptrue(P_ALL_ONE.b);
//     //     pfalse(P_ALL_ZERO.b);
//     // }
//     // if (sveLen >= SVE_256) {
//     //     ptrue(P_NOT_128.b, Xbyak_aarch64::VL16);
//     //     not_(P_NOT_128.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_128.b);
//     // }
//     // if (sveLen >= SVE_512) {
//     //     ptrue(P_NOT_256.b, Xbyak_aarch64::VL32);
//     //     not_(P_NOT_256.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_256.b);
//     // }

//     h->mov(h->X_SP, sp);
//     h->sub_imm(h->X_TRANSLATOR_STACK, h->X_SP, translator_stack_offset, h->X_TMP_0);
// }

// void postamble(dnnl::impl::cpu::aarch64::jit_generator* h) {
//     using namespace Xbyak_aarch64::util;

//     h->mov(h->x9, h->sp);

//     if (vreg_to_preserve) {
//         ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
//         ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
//     }

//     for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
//         ldp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
//                 Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
//                 post_ptr(x9, xreg_len * 2));
//     }

//     h->add(h->sp, h->sp, static_cast<int64_t>(preserved_stack_size) - 16);
//     h->ldp(h->x29, h->x30, post_ptr(h->sp, 16));
//     h->ret();
// }
} // namespace

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_power_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

    // {
    //     auto func = powf;
    //     auto res = func(2.f, 3.f);
    //     res = res + 1.f;
    //     if (res == 8.f) {
    //         std::cout << res << std::endl;
    //     }
    // }

    // {
    //     Xbyak_aarch64::XReg x8(8);

    //     //h->str(x8, pre_ptr(h->X_SP, 0x4));
    //     //h->ldr(x8, pre_ptr(h->X_SP, 0x4));

    //     //h->str(x8, reinterpret_cast<uintptr_t>(powf));
    //     // auto func_addr = reinterpret_cast<uintptr_t>(powf);
    //     auto func_addr = reinterpret_cast<uintptr_t>(my_function);
    //     h->mov(x8, func_addr);
    //     //h->ldr(x8, pre_ptr(h->X_SP, reinterpret_cast<uintptr_t>(powf)));
    //     //h->ldr(x8, pre_ptr(h->X_SP, 4));
    //     // h->ldr(x8, post_ptr(h->sp, 4));

    //     // Xbyak_aarch64::SReg s0(0);
    //     // h->fmov(s0, 2.f);

    //     // Xbyak_aarch64::SReg s1(0);
    //     // h->fmov(s1, 3.f);

    //     Xbyak_aarch64::SReg x0(0);
    //     h->fmov(x0, 2.0);

    //     Xbyak_aarch64::SReg x1(1);
    //     h->fmov(x1, 3.0);

    //     h->blr(x8);

    //     //Xbyak_aarch64::XReg x0(0);
    //     //h->ldr(x0, pre_ptr(h->X_SP, 0x4));
    //     //h->str(x0, reinterpret_cast<uintptr_t>(reinterpret_cast<uintptr_t>(powf) - 4));
    //     //h->ldr(x0, reinterpret_cast<uintptr_t>(reinterpret_cast<uintptr_t>(powf) - 4));
    // }

    // // h->mov(h->, reinterpret_cast<uintptr_t>(powf));
    // //h->b(Xbyak_aarch64::AL, loop_label);



    //h->preamble();

    // {
    //     const size_t xreg_len = 8;
    //     const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    //     const size_t vreg_to_preserve = 8; // VREG8 - VREG15
    //     const size_t num_abi_save_gpr_regs = 12;

    //     const size_t preserved_stack_size = xreg_len * (2 + num_abi_save_gpr_regs)
    //             + vreg_len_preserve * vreg_to_preserve;

    //     // const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * x0.getBit() / 8
    //     //         + vreg_to_preserve * vreg_len_preserve;

    //     using namespace Xbyak_aarch64::util;
    //     uint64_t sveLen = 0; //get_sve_length();

    //     //h->stp(h->x29, h->x30, pre_ptr(h->sp, -16));
    //     /* x29 is a frame pointer. */
    //     //h->mov(h->x29, h->sp);
    //     h->sub(h->sp, h->sp, static_cast<int64_t>(preserved_stack_size) - 16);

    //     /* x9 can be used as a temporal register. */
    //     h->mov(h->x9, h->sp);

    //     if (vreg_to_preserve) {
    //         h->st4((h->v8.d - h->v11.d)[0], post_ptr(h->x9, vreg_len_preserve * 4));
    //         h->st4((h->v12.d - h->v15.d)[0], post_ptr(h->x9, vreg_len_preserve * 4));
    //     }
    //     for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
    //         h->stp(Xbyak_aarch64::XReg(dnnl::impl::cpu::aarch64::abi_save_gpr_regs[i]),
    //                 Xbyak_aarch64::XReg(dnnl::impl::cpu::aarch64::abi_save_gpr_regs[i + 1]),
    //                 post_ptr(h->x9, xreg_len * 2));
    //     }

    //     // if (sveLen) { /* SVE is available. */
    //     //     h->ptrue(P_ALL_ONE.b);
    //     //     h->pfalse(P_ALL_ZERO.b);
    //     // }
    //     // if (sveLen >= SVE_256) {
    //     //     h->ptrue(P_NOT_128.b, Xbyak_aarch64::VL16);
    //     //     h->not_(P_NOT_128.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_128.b);
    //     // }
    //     // if (sveLen >= SVE_512) {
    //     //     h->ptrue(h->P_NOT_256.b, Xbyak_aarch64::VL32);
    //     //     h->not_(h->P_NOT_256.b, h->P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_256.b);
    //     // }

    //     h->mov(h->X_SP, h->sp);
    //     h->sub_imm(h->X_TRANSLATOR_STACK, h->X_SP, dnnl::impl::cpu::aarch64::jit_generator::translator_stack_offset, h->X_TMP_0);
    // }



    // // TODO: debug
    // std::cout << "in_vec_idxs[0]=" << in_vec_idxs[0] << std::endl;
    // std::cout << "out_vec_idxs[0]=" << out_vec_idxs[0] << std::endl;

    // using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    // TReg src = TReg(in_vec_idxs[0]);
    // TReg dst = TReg(out_vec_idxs[0]);
    // Xbyak_aarch64::XReg x8(8);

    // // preamble
    // //stp(x29, x30, pre_ptr(sp, -16));
    // //h->sub(h->sp, h->sp, static_cast<int64_t>(64 + 2 * 128));

    // auto func_addr = reinterpret_cast<uintptr_t>(my_function);
    // h->mov(x8, func_addr);

    // Xbyak_aarch64::SReg s0(0);
    // Xbyak_aarch64::SReg s1(1);
    // //h->fmov(s1, power);

    // for (auto i = 0; i < 4; i++) {
    //     h->mov(s0, src.s[i]);
    //     // TODO: move out of loop
    //     h->fmov(s1, power);
    //     h->blr(x8);

    //     Xbyak_aarch64::WReg w0(0);
    //     h->fmov(w0, s0);
    //     h->mov(dst.s[i], w0);
    // }

    // postamble
    //h->add(h->sp, h->sp, static_cast<int64_t>(64 + 2 * 128));




    // h->postamble();

    // {
    //     using namespace Xbyak_aarch64::util;

    //     h->mov(h->x9, h->sp);

    //     const size_t xreg_len = 8;
    //     const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    //     const size_t vreg_to_preserve = 8; // VREG8 - VREG15
    //     const size_t num_abi_save_gpr_regs = 12;

    //     const size_t preserved_stack_size = xreg_len * (2 + num_abi_save_gpr_regs) + vreg_len_preserve * vreg_to_preserve;

    //     if (vreg_to_preserve) {
    //         h->ld4((h->v8.d - h->v11.d)[0], post_ptr(h->x9, vreg_len_preserve * 4));
    //         h->ld4((h->v12.d - h->v15.d)[0], post_ptr(h->x9, vreg_len_preserve * 4));
    //     }

    //     for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
    //         h->ldp(Xbyak_aarch64::XReg(dnnl::impl::cpu::aarch64::abi_save_gpr_regs[i]),
    //                 Xbyak_aarch64::XReg(dnnl::impl::cpu::aarch64::abi_save_gpr_regs[i + 1]),
    //                 post_ptr(h->x9, xreg_len * 2));
    //     }

    //     h->add(h->sp, h->sp, static_cast<int64_t>(preserved_stack_size) - 16);
    //     //h->ldp(h->x29, h->x30, post_ptr(h->sp, 16));
    // }


    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    if (power == 0.f) {
        h->fmov(dst.s, 1.);
        return;
    }

    if (power == 1.f) {
        if (src.getIdx() != dst.getIdx()) {
            h->uni_orr(dst, src, src);
        }
        return;
    }

    if (std::floor(power) == power && power != 0) {
        h->fmov(dst.s, 1.);

        auto current_power = static_cast<size_t>(power);
        while (current_power > 0) {
            if (current_power & 1) {
                h->fmul(dst.s, dst.s, src.s);
            }
            if (current_power > 1) {
                h->fmul(src.s, src.s, src.s);
            }
            current_power = current_power >> 1;
        }
    } else {
        auto func_addr = reinterpret_cast<uintptr_t>(my_function);

        Xbyak_aarch64::XReg x8(8);
        h->mov(x8, func_addr);

        Xbyak_aarch64::SReg s0(0);
        Xbyak_aarch64::SReg s1(1);
        //h->fmov(s1, power);

        for (auto i = 0; i < 4; i++) {
            h->mov(s0, src.s[i]);
            // TODO: move out of loop
            h->fmov(s1, power);
            h->blr(x8);

            Xbyak_aarch64::WReg w0(0);
            h->fmov(w0, s0);
            h->mov(dst.s[i], w0);
        }
    }
}

/// RELU ///
jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node,
                                   const float alpha)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node), alpha) {
}

jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const Precision exec_prc,
                                   const float alpha)
                                   : jit_emitter(host, host_isa, exec_prc, alpha) {
}

size_t jit_relu_emitter::get_inputs_count() const { return 1; }

size_t jit_relu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32}};
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

    if (alpha != 0.f) {
        IE_THROW() << "not zero alpha is not supported";
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    h->movi(tmp.s, 0);

    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);
    h->fmaxnm(dst.s, src.s, tmp.s);
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

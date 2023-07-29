// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

// template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
// void jit_uni_eltwise_generic<isa>::generate() {
// }

// template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
// void jit_uni_eltwise_generic<isa>::generate() {
//     // TODO: not implemented

//     // auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
//     auto const exec_prc = InferenceEngine::Precision::FP32;

//     eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
//     // for (size_t i = 1; i < eltwise_data_.size(); ++i) {
//     //     post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
//     // }

//     // jit_generator::preamble
//     preamble();

//     mov(x0, x0);
//     mov(x0, x0);
//     mov(x0, x0);

//     XReg param = param1;
//     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr), X_TMP_1);
//     ldr(reg_src, ptr(X_TMP_0));

//     // XReg param = param1;
//     // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + sizeof(size_t), X_TMP_1);
//     // ldr(reg_src1, ptr(X_TMP_0));

//     // for (size_t i = 0; i < jep_.inputs_number; i++) {
//     //     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[0]) + i * sizeof(size_t), X_TMP_1);
//     //     ldr(get_src_reg(i), ptr(X_TMP_0));
//     // }

//     //mov(x0, jep_.work_amount);

//     // add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[1]), X_TMP_1);
//     // ldr(reg_src1, ptr(X_TMP_0));

//     add_imm(X_TMP_0, param, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
//     ldr(reg_dst, ptr(X_TMP_0));

//     ldr(vmm_src, ptr(reg_src));
//     //ldr(vmm_src1, ptr(get_src_reg(1)));
//     //ldr(vmm_src1, ptr(reg_src1));

//     mov(x0, x0);
//     mov(x0, x0);
//     mov(x0, x0);

//     str(vmm_src, ptr(reg_dst));

//     compute_eltwise_op();
//     mov(x0, x0);
//     mov(x0, x0);
//     mov(x0, x0);

//     // jit_generator::postamble
//     postamble();
// }

}  // namespace aarch64
}  // namespace intel_cpu
}  // namespace ov

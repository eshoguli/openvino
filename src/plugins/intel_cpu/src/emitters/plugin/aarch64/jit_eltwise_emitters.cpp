// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <memory>
#include <cmath>
#include <math.h>
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

#define OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)                        \
    OV_CPU_JIT_EMITTER_ASSERT(                                                \
        ((exec_prc_ == ov::element::f16) || (exec_prc_ == ov::element::f32)), \
        "unsupported precision: " + exec_prc_.to_string());                   \

namespace {
ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(
            input.get_source_output().get_element_type());
    }

    assert(std::all_of(
        input_precisions.begin(),
        input_precisions.end(),
        [&input_precisions](const ov::element::Type& precision) {return precision == input_precisions[0]; }));

    return input_precisions[0];
}

int float2int(const float value, const ov::element::Type& type = ov::element::f32) {
    if (type == ov::element::f16) {
        return dnnl::impl::utils::bit_cast<int16_t>(static_cast<float16>(value));
    } else if (type == ov::element::f32) {
        return dnnl::impl::utils::bit_cast<int>(value);
    } else {
        OV_CPU_JIT_EMITTER_ASSERT(false, "unsupported precision: " + type.to_string());
    }
}
} // namespace

/// ABS ///
jit_abs_emitter::jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_abs_emitter::jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_abs_emitter::get_inputs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_abs_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fabs(dst, src);
}

std::set<std::vector<element::Type>> jit_abs_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// ADD ///
jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
                                 : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_add_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fadd(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// CLAMP ///
jit_clamp_emitter::jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
                                     : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    const auto clamp = std::dynamic_pointer_cast<ov::op::v0::Clamp>(node);
    if (clamp == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to ov::op::v0::Clamp");
    }
    min = static_cast<float>(clamp->get_min());
    max = static_cast<float>(clamp->get_max());

    prepare_table();
}

jit_clamp_emitter::jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float min,
                                     const float max,
                                     const ov::element::Type exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc),
                                       min(min),
                                       max(max) {
    prepare_table();
}

size_t jit_clamp_emitter::get_inputs_count() const { return 1; }

size_t jit_clamp_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_clamp_emitter::get_aux_gprs_count() const { return 1; }

void jit_clamp_emitter::register_table_entries() {
    push_arg_entry_of("min", float2int(min, exec_prc_), true);
    push_arg_entry_of("max", float2int(max, exec_prc_), true);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_clamp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg aux = TReg(aux_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->ld1r(aux, table_val2("min"));
    h->fmax(dst, src, aux);
    h->ld1r(aux, table_val2("max"));
    h->fmin(dst, dst, aux);
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// DIVIDE ///
jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fdiv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// EQUAL ///
jit_equal_emitter::jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
                                     : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}
jit_equal_emitter::jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const ov::element::Type exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_equal_emitter::get_inputs_count() const { return 2; }

size_t jit_equal_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_equal_emitter::get_aux_gprs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
            (exec_prc_ == ov::element::f16) || (exec_prc_ == ov::element::f32),
            "unsupported precision: " + exec_prc_.to_string());
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->fcmeq(dst, src1, src2);

    h->ld1r(aux, table_val2("one"));
    h->and_(BReg(dst.getIdx()), BReg(dst.getIdx()), BReg(aux.getIdx()));
}

void jit_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true);
}

/// EXPONENT ///
jit_exp_emitter::jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_exp_emitter::jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_exp_emitter::get_inputs_count() const { return 1; }

size_t jit_exp_emitter::get_aux_vecs_count() const { return 4; }

size_t jit_exp_emitter::get_aux_gprs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_exp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(
            (exec_prc_ == ov::element::f16) || (exec_prc_ == ov::element::f32),
            "unsupported precision: " + exec_prc_.to_string());

    std::cout << "jit_exp_emitter::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);

    const TReg vmm_mask(aux_vec_idxs[3]);

    h->ld1r(vmm_aux0, table_val2("one"));

    h->ld1r(vmm_aux0, table_val2("exp_ln_flt_max_f"));
    h->fmin(vmm_dst, vmm_src, vmm_aux0);
    h->ld1r(vmm_aux0, table_val2("exp_ln_flt_min_f"));
    h->fmax(vmm_dst, vmm_dst, vmm_aux0);

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->fcmgt(vmm_mask, vmm_src, vmm_aux0);
    h->mov(BReg(vmm_aux1.getIdx()), BReg(vmm_dst.getIdx()));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->ld1r(vmm_aux0, table_val2("exp_log2ef"));
    h->ld1r(vmm_aux2, table_val2("half"));
    h->fmla(vmm_aux2, vmm_dst, vmm_aux0);

    // tmp = floorf(fx)
    h->frintm(vmm_aux2, vmm_aux2);

    // keep vmm_src = fx for further computations
    h->mov(BReg(vmm_dst.getIdx()), BReg(vmm_aux2.getIdx()));

    // x = x - fx * ln2
    h->ld1r(vmm_aux0, table_val2("ln2f"));
    h->fmls(vmm_aux1, vmm_aux2, vmm_aux0);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->ld1r(vmm_aux0, table_val2("one"));
    h->fsub(vmm_dst, vmm_dst, vmm_aux0);
    h->fcvtzs(vmm_aux2, vmm_dst);

    h->ld1r(vmm_aux0, table_val2("exponent_bias"));
    h->add(vmm_aux2, vmm_aux2, vmm_aux0);

    const int n_mantissa_bits = exec_prc_ == ov::element::f16 ? 10 : 23;
    h->sqshl(vmm_aux2, vmm_aux2, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    h->and_(BReg(vmm_aux2.getIdx()), BReg(vmm_mask.getIdx()), BReg(vmm_aux2.getIdx()));

    // compute polynomial
    h->ld1r(vmm_aux0, table_val2("exp_pol5"));
    h->ld1r(vmm_dst, table_val2("exp_pol4"));
    h->fmla(vmm_dst, vmm_aux1, vmm_aux0);

    h->ld1r(vmm_aux0, table_val2("exp_pol3"));
    h->fmla(vmm_aux0, vmm_dst, vmm_aux1);

    h->ld1r(vmm_dst, table_val2("exp_pol2"));
    h->fmla(vmm_dst, vmm_aux0, vmm_aux1);

    h->ld1r(vmm_aux0, table_val2("exp_pol1"));
    h->fmla(vmm_aux0, vmm_dst, vmm_aux1);

    h->ld1r(vmm_dst, table_val2("one"));
    h->fmla(vmm_dst, vmm_aux0, vmm_aux1);

    // y = y * 2^n
    h->fmul(vmm_dst, vmm_dst, vmm_aux2);
    h->ld1r(vmm_aux0, table_val2("two"));
    h->fmul(vmm_dst, vmm_dst, vmm_aux0);
}

void jit_exp_emitter::register_table_entries() {
//    push_arg_entry_of("exp_ln_flt_max_f", float2int(std::log(FLT_MAX), exec_prc_), true);
//    push_arg_entry_of("exp_ln_flt_min_f", float2int(std::log(FLT_MIN), exec_prc_), true);
//    push_arg_entry_of("exp_log2ef", float2int(static_cast<float>(std::log2(M_E)), exec_prc_), true);
//
//    push_arg_entry_of("one", float2int(1.f, exec_prc_), true);
//
//    push_arg_entry_of("two", float2int(2.f, exec_prc_), true);
//    push_arg_entry_of("half", float2int(0.5f, exec_prc_), true);
//    push_arg_entry_of("ln2f", float2int(std::log(2.f), exec_prc_), true);
//    //push_arg_entry_of("exponent_bias", 0x0000007f, true); //127
//    // 0x0F = 0b 0000 01111
//    // 0x1F = 0b 0000 11111
//    // 0x0000007f = 127
//    push_arg_entry_of("exponent_bias", 0x0000001f, true); //127
//    push_arg_entry_of("exp_pol1", float2int(0.999999701f, exec_prc_), true);
//    push_arg_entry_of("exp_pol2", float2int(0.499991506f, exec_prc_), true);
//    push_arg_entry_of("exp_pol3", float2int(0.166676521f, exec_prc_), true);
//    push_arg_entry_of("exp_pol4", float2int(0.0418978221f, exec_prc_), true);
//    push_arg_entry_of("exp_pol5", float2int(0.00828929059f, exec_prc_), true);


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

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    //return {{element::f16}, {element::f32}};
    return {{element::f32}};
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
                                         : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc)
                                         : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_mul_add_emitter::get_inputs_count() const { return 3; }

size_t jit_mul_add_emitter::get_aux_vecs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg dst = TReg(out_vec_idxs[0]);

    TReg mul0(in_vec_idxs[0]);
    if (dst.getIdx() == in_vec_idxs[0]) {
        //TReg aux(aux_vec_idxs[0]);
        //TReg src0(in_vec_idxs[0]);
        h->mov(BReg(aux_vec_idxs[0]), BReg(in_vec_idxs[0]));
        mul0 = TReg(aux_vec_idxs[0]);
    }

    TReg mul1(in_vec_idxs[1]);
    if (dst.getIdx() == in_vec_idxs[1]) {
        //TReg aux(aux_vec_idxs[0]);
        //TReg src1(in_vec_idxs[1]);
        h->mov(BReg(aux_vec_idxs[0]), BReg(in_vec_idxs[1]));
        mul1 = TReg(aux_vec_idxs[0]);
    }

    if (dst.getIdx() != in_vec_idxs[2]) {
        //TReg src2(in_vec_idxs[2]);
        h->mov(BReg(dst.getIdx()), BReg(in_vec_idxs[2]));
    }

    h->fmla(dst, mul0, mul1);
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16, element::f16}, {element::f32, element::f32, element::f32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fmul(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// POWER ///
jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const std::shared_ptr<ov::Node>& node,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, node, exec_prc) {
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to snippets::op::PowerStatic");
    }

    power = powerStaticNode->get_power();
    scale = 1.f;
    shift = 0.f;

    prepare_table();
}

jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const float power,
                                                   const float scale,
                                                   const float shift,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, exec_prc),
                                                     power(power),
                                                     scale(scale),
                                                     shift(shift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_gprs_count() const { return 2; }

void jit_power_static_emitter::register_table_entries() {
    push_arg_entry_of("power", float2int(power, exec_prc_), true);
    push_arg_entry_of("scale", float2int(scale, exec_prc_), true);
    push_arg_entry_of("shift", float2int(shift, exec_prc_), true);
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    //return {{element::f16}, {element::f32}};
    return {{element::f32}};
}

extern "C" {
float my_powf(float value1, float value2) {
    std::cout << "value1: " << value1 << ", value2: " << value2 << std::endl;
    return ::powf(value1, value2);
}
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);

    //std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;
    std::cout << "power: " << power << ", scale: " << scale << ", shift: " << shift << std::endl;

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg; // vector: 32 bytes or 16 bytes
    //using SReg = typename cpu_isa_vector_traits<isa, type>::SReg; // scalar: 32 bytes or 16 bytes
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    TReg dst = TReg(out_vec_idxs[0]);

    if (power == 0.f) {
        h->fmov(dst, 1.);
        return;
    }

    bool get_from_dst = false;
    const auto src = [&in_vec_idxs, &out_vec_idxs, &get_from_dst]() -> TReg {
        return get_from_dst ? TReg(out_vec_idxs[0]) : TReg(in_vec_idxs[0]);
    };

    TReg aux = TReg(aux_vec_idxs[0]);
    if (scale != 1.f) {
        auto adr = table_val2("scale");
        h->ld1r(aux, adr);
        h->fmul(dst, src(), aux);
        get_from_dst = true;
    }

    if (shift != 0.f) {
        auto adr = table_val2("shift");
        h->ld1r(aux, adr);
        h->fadd(dst, src(), aux);
        get_from_dst = true;
    }

    if (power == 1.f) {
        if (!get_from_dst && (in_vec_idxs[0] != dst.getIdx())) {
            h->mov(BReg(dst.getIdx()), BReg(src().getIdx()));
        }
        return;
    }

    if (std::floor(power) == power && power > 0) {
        h->mov(BReg(aux.getIdx()), BReg(src().getIdx()));
        h->fmov(dst, 1.);

        auto current_power = static_cast<size_t>(power);
        while (current_power > 0) {
            if (current_power & 1) {
                h->fmul(dst, dst, aux);
            }
            if (current_power > 1) {
                h->fmul(aux, aux, aux);
            }
            current_power = current_power >> 1;
        }
    } else {
        std::cout << "jit_power_static_emitter::emit_isa" << std::endl;
        auto pow_f32_addr = reinterpret_cast<uintptr_t>(::powf);
        //auto pow_f32_addr = reinterpret_cast<uintptr_t>(my_powf);

        Xbyak_aarch64::XReg func_reg(aux_gpr_idxs[0]);
        h->mov(func_reg, pow_f32_addr);

        Xbyak_aarch64::SReg s0(0);
        Xbyak_aarch64::SReg s1(1);

        const std::unordered_set<size_t> exclude = {src().getIdx(), dst.getIdx()};
        store_context(exclude);
        const auto length = exec_prc_ == ov::element::f32 ? 4 : 8;
        for (auto i = 0; i < length; i++) {
            if (exec_prc_ == ov::element::f32) {
                // TODO: not completed
                //h->mov(s0, src()[i]);
                Xbyak_aarch64::VReg4S src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                h->mov(s0, src2[i]);
                h->ldr(s1, table_val("power"));
            } else if (exec_prc_ == ov::element::f16) {
                Xbyak_aarch64::HReg h0_16_scalar(0);
                Xbyak_aarch64::HReg h1_16_scalar(1);

                // TODO: not completed: fp32 is commented
                Xbyak_aarch64::VReg8H src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                h->mov(h0_16_scalar, src2[i]);
                h->fcvt(s0, h0_16_scalar);

                h->ldr(h1_16_scalar, table_val("power"));
                h->fcvt(s1, h1_16_scalar);
            }

            h->str(Xbyak_aarch64::QReg(dst.getIdx()), pre_ptr(h->sp, -16));
            h->str(Xbyak_aarch64::QReg(src().getIdx()), pre_ptr(h->sp, -16));
            h->blr(func_reg);
            h->ldr(Xbyak_aarch64::QReg(src().getIdx()), post_ptr(h->sp, 16));
            h->ldr(Xbyak_aarch64::QReg(dst.getIdx()), post_ptr(h->sp, 16));

            if (exec_prc_ == ov::element::f32) {
                Xbyak_aarch64::WReg w0(0);
                h->fmov(w0, s0);
                h->mov(dst[i], w0);
            } else if (exec_prc_ == ov::element::f16) {
                Xbyak_aarch64::HReg h0_16_scalar(0);
                Xbyak_aarch64::HReg h1_16_scalar(1);

                // TODO: rework
                h->fcvt(h0_16_scalar, s0);
                Xbyak_aarch64::WReg w0(0);
                h->fmov(w0, h0_16_scalar);
                h->mov(dst[i], w0);
            }
        }
        restore_context(exclude);
    }
}

/// PRELU ///
jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_prelu_emitter::get_inputs_count() const { return 2; }

size_t jit_prelu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fcmge(dst.s, src1.s, 0.0);
    h->fmul(tmp.s, src1.s, src2.s);
    h->bsl(dst.b16, src1.b16, tmp.b16);
}

/// RELU ///
jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_relu_emitter::get_inputs_count() const { return 1; }

size_t jit_relu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->movi(tmp.s, 0);
    h->fmaxnm(dst.s, src.s, tmp.s);
}

/// SELECT ///
jit_select_emitter::jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
                                       : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
}
jit_select_emitter::jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const ov::element::Type exec_prc)
                                       : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_select_emitter::get_inputs_count() const { return 3; }

size_t jit_select_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_select_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

void jit_select_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_select_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg src3 = TReg(in_vec_idxs[2]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->eor(aux.b16, aux.b16, aux.b16);
    h->fcmgt(aux.s, src1.s, aux.s);

    h->bsl(aux.b16, src2.b16, src3.b16);
    h->mov(dst.b16, aux.b16);
}

/// SIGMOID ///
jit_sigmoid_emitter::jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, node);
}

jit_sigmoid_emitter::jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_sigmoid_emitter::get_inputs_count() const { return 1; }

size_t jit_sigmoid_emitter::get_aux_vecs_count() const {
    return exp_emitter->get_aux_vecs_count() + 2;
}

size_t jit_sigmoid_emitter::get_aux_gprs_count() const {
    return exp_emitter->get_aux_gprs_count() + 1;
}

void jit_sigmoid_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_sigmoid_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);

    const TReg vmm_aux0(aux_vec_idxs[exp_emitter->get_aux_vecs_count() + 1]);
    const TReg vmm_mask(aux_vec_idxs[exp_emitter->get_aux_vecs_count()]);

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_mask for the mask as exp_compute does not use it.
    // we store the original sign and make x negative
    h->eor(vmm_aux0.b16, vmm_aux0.b16, vmm_aux0.b16);
    h->fcmgt(vmm_mask.s, vmm_src.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table_val2("sign_mask"));
    h->orr(vmm_aux0.b16, vmm_src.b16, vmm_aux0.b16);

    exp_emitter->emit_code(
            { vmm_aux0.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    // dup exp(x)
    h->mov(vmm_aux1.b16, vmm_dst.b16);
    // (exp(x) + 1)
    h->ld1r(vmm_aux0.s, table_val2("one"));
    h->fadd(vmm_aux1.s, vmm_aux1.s, vmm_aux0.s);
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_dst.s, vmm_dst.s, vmm_aux1.s);

    // Now we have to apply the "symmetry" based on original sign
    h->ld1r(vmm_aux2.s, table_val2("one"));
    h->fsub(vmm_aux2.s, vmm_aux2.s, vmm_dst.s);

    h->bsl(vmm_mask.b16, vmm_aux2.b16, vmm_dst.b16);
    h->mov(vmm_dst.b16, vmm_mask.b16);
}

void jit_sigmoid_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("sign_mask", 0x80000000, true);
}

void jit_sigmoid_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_sigmoid_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// SUBTRACT ///
jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_subtract_emitter::get_inputs_count() const { return 2; }

void jit_subtract_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);
    std::cout << __PRETTY_FUNCTION__ << "::emit_isa: " << exec_prc_ << std::endl;

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fsub(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// SWISH ///
jit_swish_emitter::jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, node);
}

jit_swish_emitter::jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float beta,
                                     const ov::element::Type exec_prc)
        : jit_emitter(host, host_isa, exec_prc), beta(beta) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, exec_prc);
}

size_t jit_swish_emitter::get_inputs_count() const {return 1; }

size_t jit_swish_emitter::get_aux_vecs_count() const {
    return sigmoid_emitter->get_aux_vecs_count() + 2;
}

size_t jit_swish_emitter::get_aux_gprs_count() const {
    return sigmoid_emitter->get_aux_gprs_count() + 1;
}

void jit_swish_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_swish_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_orig_src(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count()]);
    const TReg vmm_aux(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count() + 1]);

    h->mov(vmm_orig_src.b16, vmm_src.b16);

    // x*beta
    h->ld1r(vmm_aux.s, table_val2("beta"));
    h->fmul(vmm_aux.s, vmm_aux.s, vmm_src.s);

    // sigmoid(x*beta)
    sigmoid_emitter->emit_code(
            { vmm_aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    // x*sigmoid(x*beta)
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_orig_src.s);
}

void jit_swish_emitter::register_table_entries() {
    push_arg_entry_of("beta", dnnl::impl::float2int(beta), true);
}

void jit_swish_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_swish_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// TANH ///
jit_tanh_emitter::jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, node);
}

jit_tanh_emitter::jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, exec_prc);
}

size_t jit_tanh_emitter::get_inputs_count() const { return 1; }

size_t jit_tanh_emitter::get_aux_vecs_count() const {
    return sigmoid_emitter->get_aux_vecs_count() + 1;
}

size_t jit_tanh_emitter::get_aux_gprs_count() const {
    return sigmoid_emitter->get_aux_gprs_count() + 1;
}

void jit_tanh_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_tanh_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    TReg aux = TReg(aux_vec_idxs.back());

    h->ld1r(aux.s, table_val2("two"));
    h->uni_fmul(aux.s, src.s, aux.s);

    sigmoid_emitter->emit_code(
            { aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    h->ld1r(aux.s, table_val2("two"));
    h->uni_fmul(dst.s, aux.s, dst.s);
    h->ld1r(aux.s, table_val2("one"));
    h->uni_fsub(dst.s, dst.s, aux.s);
}

void jit_tanh_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("two", 0x40000000, true);
}

void jit_tanh_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_tanh_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

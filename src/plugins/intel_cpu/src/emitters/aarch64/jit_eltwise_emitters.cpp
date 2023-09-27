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
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        IE_THROW() << "Can't cast to snippets::op::PowerStatic";
    }

    // scale = 1.f;
    // shift = 0.f;

    prepare_table();
}

jit_power_emitter::jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float power,
                                     const Precision exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc), power(power) {
    prepare_table();
}

size_t jit_power_emitter::get_inputs_count() const { return 1; }

size_t jit_power_emitter::get_aux_vecs_count() const { return 2; }

size_t jit_power_emitter::get_aux_gprs_count() const { return 1; }

void jit_power_emitter::register_table_entries() {
    push_arg_entry_of("power", dnnl::impl::float2int(power), true);
    // push_arg_entry_of("scale", float2int(scale), true);
    // push_arg_entry_of("shift", float2int(shift), true);
    // push_arg_entry_of("one",   float2int(1.f), true);
}

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

namespace {
extern "C" float my_function(float v1, float v2);
float pow_f32(float v1, float v2) {
    return pow(v1, v2);
}
} // namespace

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_power_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision: " << exec_prc_;
    }

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

    if (std::floor(power) == power && power > 0) {
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
        auto pow_f32_addr = reinterpret_cast<uintptr_t>(pow_f32);

        Xbyak_aarch64::XReg x8(8);
        h->mov(x8, pow_f32_addr);

        Xbyak_aarch64::SReg s0(0);
        Xbyak_aarch64::SReg s1(1);

        for (auto i = 0; i < 4; i++) {
            h->mov(s0, src.s[i]);

            //const float power2 = 1.23;
            //h->fmov(s1, power2);
            h->ldr(s1, table_val("power"));

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

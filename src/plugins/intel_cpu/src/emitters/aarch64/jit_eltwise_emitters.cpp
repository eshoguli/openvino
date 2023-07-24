// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <memory>
#include "ie_ngraph_utils.hpp"

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
                                 const std::shared_ptr<ov::Node>& node)
                                 : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 Precision exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_add_emitter::get_inputs_num() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::sve_512) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_512>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_384) {
        // TODO: not supported
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_256) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_128) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_128>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision";
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
                                         const std::shared_ptr<ov::Node>& node)
                                         : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         Precision exec_prc)
                                         : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_mul_add_emitter::get_inputs_num() const { return 3; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::sve_512) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_512>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_384) {
        // TODO: not supported
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_256) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_128) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_128>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision";
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
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           Precision exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_num() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::sve_512) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_512>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_384) {
        // TODO: not supported
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_256) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_128) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_128>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != Precision::FP32) {
        IE_THROW() << "unsupported precision";
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
                                     const std::shared_ptr<ov::Node>& node,
                                     Precision exec_prc)
                                     : jit_emitter(host, host_isa, node, exec_prc) {
}

jit_power_emitter::jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     Precision exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_power_emitter::get_inputs_num() const { return 2; }

size_t jit_power_emitter::aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_power_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_power_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::sve_512) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_512>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_384) {
        // TODO: not supported
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_256) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_256>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::sve_128) {
        emit_isa<dnnl::impl::cpu::aarch64::sve_128>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_power_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    // TODO: uncomment when not supported instruction set will be removed
    //using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    // TODO: debug
    using TReg = Xbyak_aarch64::VReg;

    // TODO: if the second input is scalar then it can be hardcoded and easily unrolled
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);

    TReg exp = TReg(aux_vec_idxs[0]);
    h->uni_fcvtzs(exp.s, src1.s);
    Xbyak_aarch64::WReg counter{aux_gpr_idxs[0]};
    h->mov(counter, exp.s[0]);

    TReg dst = TReg(out_vec_idxs[0]);
    h->uni_orr(dst, src0, src0);

    Xbyak_aarch64::Label loop_label;
    Xbyak_aarch64::Label loop_end_label;
    h->L(loop_label);
    {
        // TODO: unroll
        h->cmp(counter, 2);
        h->b(Xbyak_aarch64::LO, loop_end_label);

        h->uni_fmul(dst.s, dst.s, src0.s);

        h->sub(counter, counter, 1);
        h->b(Xbyak_aarch64::AL, loop_label);
    }
    h->L(loop_end_label);
}

jit_dnnl_emitter::jit_dnnl_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           dnnl_alg_kind_t alg_kind,
                                           float alpha,
                                           float beta,
                                           InferenceEngine::Precision exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        // eltwise_injector_asimd = std::make_shared<dnnl::impl::cpu::aarch64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::aarch64::asimd>>(
        //     h,
        //     alg_kind,
        //     alpha,
        //     beta,
        //     1.f);
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }
}

size_t jit_dnnl_emitter::get_inputs_num() const {
    return 1;
}

void jit_dnnl_emitter::emit_impl(
    const std::vector<size_t>& in_vec_idxs,
    const std::vector<size_t>& out_vec_idxs) const {
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_dnnl_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    // using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    // using TRegS = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TRegS;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"
#include "ie_ngraph_utils.hpp"

#define CONST_1_F 0x3f800000 // 1.f
#define INF_MASK  0x7F800000
#define INF_NEG_MASK 0xFF800000

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;

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
jit_add_emitter::jit_add_emitter(
    dnnl::impl::cpu::aarch64::jit_generator *host,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& node) : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_add_emitter::jit_add_emitter(
    dnnl::impl::cpu::aarch64::jit_generator *host,
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
    using Vmm = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    h->uni_fadd(vmm_dst.s, vmm_src0.s, vmm_src1.s);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(
    dnnl::impl::cpu::aarch64::jit_generator *host,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& node) : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(
    dnnl::impl::cpu::aarch64::jit_generator *host,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    Precision exec_prc) : jit_emitter(host, host_isa, exec_prc) {}

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
    // using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    // Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    // Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    // Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    // auto uni_vmul = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
    //     switch (exec_prc_) {
    //         case Precision::FP32: h->uni_vmulps(vmm_dst, vmm_src0, vmm_src1); break;
    //         case Precision::I32:  h->uni_vpmulld(vmm_dst, vmm_src0, vmm_src1); break;
    //         default: assert(!"unsupported precision");
    //     }
    // };

    // if (isa == x64::sse41) {
    //     h->uni_vmovups(vmm_dst, vmm_src0);
    //     uni_vmul(vmm_dst, vmm_dst, vmm_src1);
    // } else {
    //     uni_vmul(vmm_dst, vmm_src0, vmm_src1);
    // }
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

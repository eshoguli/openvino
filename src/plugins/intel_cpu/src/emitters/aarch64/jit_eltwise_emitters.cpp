// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"
#include "ie_ngraph_utils.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak;

#define CONST_1_F 0x3f800000 // 1.f
#define INF_MASK  0x7F800000
#define INF_NEG_MASK 0xFF800000

namespace ov {
namespace intel_cpu {
namespace aarch64 {

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
jit_add_emitter::jit_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}
jit_add_emitter::jit_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, Precision exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_add_emitter::get_inputs_num() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

template <x64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vadd = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case Precision::FP32: h->uni_vaddps(vmm_dst, vmm_src0, vmm_src1); break;
            case Precision::I32:  h->uni_vpaddd(vmm_dst, vmm_src0, vmm_src1); break;
            default: assert(!"unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vadd(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vadd(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}
jit_mul_add_emitter::jit_mul_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, Precision exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_mul_add_emitter::get_inputs_num() const { return 3; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

template <x64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_src2 = Vmm(in_vec_idxs[2]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vfmadd231_xmm = [this](Xmm vmm_dst, Xmm vmm_src0, Xmm vmm_src1, Xmm vmm_src2) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        switch (exec_prc_) {
            case Precision::FP32: {
                h->uni_vmulps(vmm_dst, vmm_dst, vmm_src1);
                h->uni_vaddps(vmm_dst, vmm_dst, vmm_src2);
            } break;
            case Precision::I32: {
                h->uni_vpmulld(vmm_dst, vmm_dst, vmm_src1);
                h->uni_vpaddd(vmm_dst, vmm_dst, vmm_src2);
            } break;
            default: assert(!"unsupported precision");
        }
    };

    auto uni_vfmadd231_vmm = [this, vmm_aux0](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1, Vmm vmm_src2) {
        switch (exec_prc_) {
            case Precision::FP32: {
                Vmm vmm_mul0;
                if (vmm_dst.getIdx() == vmm_src0.getIdx()) {
                    h->uni_vmovups(vmm_aux0, vmm_src0);
                    vmm_mul0 = vmm_aux0;
                } else {
                    vmm_mul0 = vmm_src0;
                }

                Vmm vmm_mul1;
                if (vmm_dst.getIdx() == vmm_src1.getIdx()) {
                    h->uni_vmovups(vmm_aux0, vmm_src1);
                    vmm_mul1 = vmm_aux0;
                } else {
                    vmm_mul1 = vmm_src1;
                }

                if (vmm_dst.getIdx() != vmm_src2.getIdx())
                    h->uni_vmovups(vmm_dst, vmm_src2);

                h->uni_vfmadd231ps(vmm_dst, vmm_mul0, vmm_mul1);
            } break;
            case Precision::I32: {
                h->uni_vpmulld(vmm_dst, vmm_src0, vmm_src1);
                h->uni_vpaddd(vmm_dst, vmm_dst, vmm_src2);
            } break;
            default: assert(!"unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        uni_vfmadd231_xmm(vmm_dst, vmm_src0, vmm_src1, vmm_src2);
    } else {
        uni_vfmadd231_vmm(vmm_dst, vmm_src0, vmm_src1, vmm_src2);
    }
}

size_t jit_mul_add_emitter::aux_vecs_count() const {
    return 1;
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32, element::f32}, {element::i32, element::i32, element::i32}};
}

/// SUB ///
jit_subtract_emitter::jit_subtract_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}
jit_subtract_emitter::jit_subtract_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, Precision exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_subtract_emitter::get_inputs_num() const { return 2; }

void jit_subtract_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

template <x64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vsub = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case Precision::FP32: h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1); break;
            case Precision::I32:  h->uni_vpsubd(vmm_dst, vmm_src0, vmm_src1); break;
            default: assert(!"unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vsub(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vsub(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}
jit_multiply_emitter::jit_multiply_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, Precision exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_num() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

template <x64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vmul = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case Precision::FP32: h->uni_vmulps(vmm_dst, vmm_src0, vmm_src1); break;
            case Precision::I32:  h->uni_vpmulld(vmm_dst, vmm_src0, vmm_src1); break;
            default: assert(!"unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vmul(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vmul(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// DIVIDE ///
jit_divide_emitter::jit_divide_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, Precision exec_prc)
: jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}
jit_divide_emitter::jit_divide_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, Precision exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_num() const { return 2; }

void jit_divide_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

template <x64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vdiv = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case Precision::FP32: {
                h->uni_vdivps(vmm_dst, vmm_src0, vmm_src1);
                break;
            }
            case Precision::I32: {
                Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);

                // The opset doesn't contain vector instruction for integer divide operation
                // As WA we emulate its behavior via fp divide followed by rounding to zero
                h->uni_vcvtdq2ps(vmm_dst, vmm_src0);
                h->uni_vcvtdq2ps(vmm_aux0, vmm_src1);
                h->uni_vdivps(vmm_dst, vmm_dst, vmm_aux0);
                h->uni_vroundps(vmm_dst, vmm_dst, 3); // rounding to zero
                h->uni_vcvtps2dq(vmm_dst, vmm_dst);
                break;
            }
            default: assert(!"unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vdiv(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vdiv(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

size_t jit_divide_emitter::aux_vecs_count() const {
    return exec_prc_ == Precision::I32 ? 1 : 0;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

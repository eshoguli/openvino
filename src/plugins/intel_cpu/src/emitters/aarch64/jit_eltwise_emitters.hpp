// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "jit_injector.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                    const float alpha = 0.f);

    jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& node,
                    const float alpha = 0.f);

    size_t get_inputs_num() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                        const float alpha = 0.f);

    jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& node,
                        const float alpha = 0.f);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                         const float alpha = 0.f);

    jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         const std::shared_ptr<ov::Node>& node,
                         const float alpha = 0.f);

    size_t get_inputs_num() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_power_emitter : public jit_emitter {
public:
    jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                      const float alpha = 0.f);

    jit_power_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node,
                      const float alpha = 0.f);

    size_t get_inputs_num() const override;

    size_t aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_relu_emitter : public jit_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                     const float alpha = 0.f);

    jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& node,
                     const float alpha = 0.f);

    size_t get_inputs_num() const override;

    size_t aux_vecs_count() const override;

    // TODO: debug
    // void emit_data() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    std::shared_ptr<jit_injector_f32<dnnl::impl::cpu::aarch64::asimd>> jit_injector;
};

class jit_exp_emitter : public jit_emitter {
public:
    jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                      const float alpha = 0.f);

    jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& node,
                      const float alpha = 0.f);

    size_t get_inputs_num() const override;

    size_t aux_vecs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_dnnl_emitter : public jit_emitter {
public:
    jit_dnnl_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                         dnnl_alg_kind_t alg_kind,
                         float alpha,
                         float beta,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    std::shared_ptr<dnnl::impl::cpu::aarch64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::aarch64::asimd>> eltwise_injector_asimd;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

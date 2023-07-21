// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>

#include <emitters/aarch64/jit_emitter.hpp>
#include <emitters/aarch64/jit_eltwise_emitters.hpp>

#include <onednn/dnnl.h>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include "nodes/kernels/jit_eltwise_call_args_ptrs.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

// TODO: not completed
#define MAX_ELTWISE_INPUTS 7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
    bool use_runtime_ptrs;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

struct jit_uni_eltwise_kernel {
    void (*ker_)(const node::jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const node::jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);

        // TODO: debug only
        // const auto src_ptr = static_cast<const float*>(const_args->src_ptr[0]);
        // std::cout << "jit_uni_eltwise_kernel::operator(), src_ptr: " << std::endl;
        // for (size_t i = 0; i < 8; i++) {
        //     std::cout << src_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

struct EltwiseData {
    Algorithm algo;
    dnnl::algorithm onednnAlgorithm;
    float alpha;
    float beta;
    float gamma;

    bool operator==(const EltwiseData& rhs) const noexcept {
        return algo == rhs.algo &&
            onednnAlgorithm == rhs.onednnAlgorithm &&
            alpha == rhs.alpha &&
            beta == rhs.beta &&
            gamma == rhs.gamma;
    }
};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                     const std::vector<EltwiseData>& eltwise_data,
                                     const std::vector<ov::intel_cpu::Type>& ops_list,
                                     const dnnl::post_ops& post_ops) :
                                     jit_uni_eltwise_kernel(jep),
                                     jit_generator(),
                                     eltwise_data_(eltwise_data),
                                     ops_list_(ops_list),
                                     post_ops_(post_ops) {}

    jit_uni_eltwise_generic() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto get_precision = []() {
            const InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32;
            return exec_prc;
        };

        const auto exec_prc = get_precision();

        eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);

        preamble();

        XReg param2 = abi_param2;
        const int offset_count = jep_.input_size - 1;

        auto init_ptrs_with_offsets = [this, offset_count, param2](XReg pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    XReg offset_reg(14);
                    mov(offset_reg, offsets[j]); // x14 <= 16

                    // TODO: do we really need X_TMP_0?
                    add_imm(X_TMP_0, param2, j * sizeof(size_t), X_TMP_1); // x23 <= param2 + 16
                    XReg index_reg(15);
                    // x15 <= ptr(param2 + 16):
                    // iter #1: index_reg = 0
                    // iter #2: index_reg = 1
                    ldr(index_reg, ptr(X_TMP_0));

                    mul(offset_reg, offset_reg, index_reg); // x14 = param2
                    add(pointer, pointer, offset_reg);

                    // TODO: debug only
                    //std::cout << offsets[j] << " x " << j * sizeof(size_t) << std::endl;
                }
            }
        };

        for (size_t i = 0; i < jep_.inputs_number; i++) {
            // TODO: do we really need X_TMP_0?
            add_imm(X_TMP_0, param1, offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t), X_TMP_1);
            ldr(get_src_reg(i), ptr(X_TMP_0));

            init_ptrs_with_offsets(get_src_reg(i), jep_.src_offsets[i]);
        }

        add_imm(X_TMP_0, param1, offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr), X_TMP_1);
        ldr(reg_dst, ptr(X_TMP_0));
        init_ptrs_with_offsets(reg_dst, jep_.dst_offsets);

        mov(reg_work_amount, jep_.work_amount);

        uni_ld1rw(get_vmm_reg(1).s4, get_src_reg(1), 0);

        Label main_loop_label;
        Label main_loop_end_label;
        L(main_loop_label);
        {
            const size_t vlen = cpu_isa_traits<isa>::vlen;
            const size_t exec_prc_size = exec_prc.size();
            const size_t loop_step = vlen / exec_prc_size;

            cmp(reg_work_amount, 0x0);
            b(EQ, main_loop_end_label);

            const auto offset = jep_.dst_prc.size() * loop_step;

            uni_ldr(get_vmm_reg(0), get_src_reg(0));
            add(get_src_reg(0), get_src_reg(0), offset);

            compute_eltwise_op();

            uni_str(vmm_dst, reg_dst);


            add(reg_dst, reg_dst, offset);

            sub(reg_work_amount, reg_work_amount, loop_step);

            b(AL, main_loop_label);
        }
        L(main_loop_end_label);

        compute_eltwise_op();

        postamble();
    }

private:
    //using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    // TODO: aarch64::sve_384 is not supported
    // TODO: SIMD & FP scalar register is used
    //using TReg = QReg;

    // TODO: aarch64::sve_384 is not supported
    // TODO: SVE SIMD Vector Register is used explicitly
    //using TReg = ZRegQ;
    using TReg = VReg;

    const XReg reg_work_amount = x9;
    Xbyak_aarch64::XReg reg_dst = x10;

    Xbyak_aarch64::VReg4S xmm_src {1};
    TReg vmm_src0 {1};
    TReg vmm_src1 {2};
    TReg vmm_dst {9};

    struct EltwiseEmitterContext {
        std::shared_ptr<jit_emitter> emitter;
        dnnl::impl::cpu::aarch64::jit_generator *host;
        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa;
        const EltwiseData& opData;
        InferenceEngine::Precision exec_prc;
    };

    template<typename T>
    struct EltwiseEmitter {
        void operator()(EltwiseEmitterContext & ctx) {
            ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
        }
    };

    XReg get_src_reg(int idx) {
        return XReg(11 + idx);
    }

    TReg get_vmm_reg(int idx) {
        return TReg(1 + idx);
    }

    TReg get_aux_vmm(int idx) {
        return TReg(10 + idx);
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, InferenceEngine::Precision exec_prec) {
        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            data,
            exec_prec
        };

        OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
        OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::aarch64::jit_add_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, ov::intel_cpu::aarch64::jit_multiply_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }

    inline void compute_eltwise_op() {
        std::vector<size_t> in_idxs;
        std::vector<size_t> aux_idxs;
        for (size_t i = 0; i < eltwise_emitter->get_inputs_num(); i++)
            in_idxs.push_back(get_vmm_reg(i).getIdx());
        for (size_t i = 0; i < eltwise_emitter->aux_vecs_count(); i++)
            aux_idxs.push_back(get_aux_vmm(i).getIdx());

        std::vector<size_t> out_idxs;
        out_idxs.push_back(vmm_dst.getIdx());

        eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs);
    }

    const std::vector<EltwiseData>& eltwise_data_;
    const std::vector<ov::intel_cpu::Type>& ops_list_;
    const dnnl::post_ops& post_ops_;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov

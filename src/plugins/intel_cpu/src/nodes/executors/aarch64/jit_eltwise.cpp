// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise.hpp"
#include <vector>

namespace ov {
namespace intel_cpu {
namespace executors {
namespace aarch64 {

bool JitEltwiseExecutor::isSupported(
    const Node* node,
    const float alpha,
    const float beta,
    const float gamma) {
    const Algorithm& algorithm = node->getAlgorithm();
    const auto is_supported = one_of(algorithm,
                                    Algorithm::EltwiseAdd,
                                    Algorithm::EltwiseMultiply,
                                    Algorithm::EltwiseMulAdd,
                                    Algorithm::EltwisePowerStatic,
                                    Algorithm::EltwiseRelu);
    if (!is_supported) {
        return false;
    }

    {
        const auto& input_precisions = node->getOriginalInputPrecisions();
        if (std::any_of(input_precisions.begin(),
                        input_precisions.end(),
                        [](const InferenceEngine::Precision& precision) { return precision != InferenceEngine::Precision::FP32; })) {
            return false;
        }
        for (size_t i = 0; i < input_precisions.size(); ++i) {
            if (node->getInputShapeAtPort(i).isDynamic()) {
                return false;
            }
        }
    }

    {
        const auto& output_precisions = node->getOriginalOutputPrecisions();
        if (std::any_of(output_precisions.begin(),
                        output_precisions.end(),
                        [](const InferenceEngine::Precision& precision) { return precision != InferenceEngine::Precision::FP32; })) {
            return false;
        }
        for (size_t i = 0; i < output_precisions.size(); ++i) {
            if (node->getOutputShapeAtPort(i).isDynamic()) {
                return false;
            }
        }
    }

    if ((algorithm == Algorithm::EltwiseRelu) && ((alpha != 0.f) || (beta != 0.f) || (gamma != 0.f))) {
        return false;
    }

    if ((algorithm == Algorithm::EltwisePowerStatic) && ((beta != 1.f) || (gamma != 0.f))) {
        return false;
    }

    return true;
}

JitEltwiseExecutor::JitEltwiseExecutor(const ExecutorContext::CPtr context) : EltwiseExecutor(context) {}

bool JitEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs,
                              const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    return true;
}

void JitEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src,
                              const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    exec_func();
}

}   // namespace aarch64
}   // namespace executors
}   // namespace intel_cpu
}   // namespace ov

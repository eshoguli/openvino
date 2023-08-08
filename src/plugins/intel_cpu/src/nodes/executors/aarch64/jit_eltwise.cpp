// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise.hpp"

#include <vector>

namespace ov {
namespace intel_cpu {
namespace executors {
namespace aarch64 {

bool JitEltwiseExecutor::isSupported(const Algorithm& algorithm,
                                     const std::vector<Shape>& input_shapes,
                                     const std::vector<Shape>& outputShapes) {
    const auto is_supported = one_of(algorithm,
                                    Algorithm::EltwiseAdd,
                                    Algorithm::EltwiseMultiply,
                                    Algorithm::EltwiseMulAdd);
                                    //Algorithm::EltwisePowerDynamic);
    if (!is_supported) {
        return false;
    }

    // TODO: not completed
    // if (algorithm == Algorithm::EltwisePowerDynamic) {
    //     // TODO: fuse?
    //     if ((input_shapes.size() != 2) || (input_shapes[1].isDynamic()) || (input_shapes[1].getElementsCount() != 1)) {
    //         return false;
    //     }
    // }

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

bool JitEltwiseExecutorBuilder::isSupported(const EltwiseAttrs& eltwiseAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs) const {
    auto checkPrecision = [&srcDescs, &dstDescs](std::vector<Precision> srcVecPrc, Precision dstPrc) -> bool {
        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (srcDescs[i]->getPrecision() != srcVecPrc[i]) return false;
        }
        if (dstDescs[0]->getPrecision() != dstPrc) { return false; }
        return true;
    };

    switch (eltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
        case Algorithm::EltwiseMultiply:
        case Algorithm::EltwiseMulAdd:
        //case Algorithm::EltwisePowerDynamic:
            if (!checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32)) {
                return false;
            }
            break;
        default:
            return false;
    }

    return true;
}

}   // namespace aarch64
}   // namespace executors
}   // namespace intel_cpu
}   // namespace ov

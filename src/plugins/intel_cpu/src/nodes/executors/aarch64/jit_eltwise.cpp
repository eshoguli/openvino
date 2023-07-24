// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise.hpp"

#include <vector>

namespace ov {
namespace intel_cpu {
namespace executors {
namespace aarch64 {

bool JitEltwiseExecutor::isSupported(const Algorithm& algorithm) {
    // TODO: should we check precision here?
    const auto is_supported = one_of(algorithm,
                                    Algorithm::EltwiseAdd,
                                    Algorithm::EltwiseMultiply,
                                    Algorithm::EltwiseMulAdd,
                                    // TODO: debug: temporary uncommented: CPU tests don't support enabled case
                                    Algorithm::EltwisePowerDynamic,
                                    Algorithm::EltwisePowerStatic);
                                    // TODO: debug: PRelu is not implemented
                                    //Algorithm::EltwiseRelu);
                                    // TODO: debug: wip
                                    //Algorithm::EltwiseExp);
    if (!is_supported) {
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

    // TODO: should we check precision here?
    switch (eltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
        case Algorithm::EltwiseMultiply:
        case Algorithm::EltwiseMulAdd:
        // TODO: debug: temporary uncommented: CPU tests don't support enabled case
        case Algorithm::EltwisePowerStatic:
        case Algorithm::EltwisePowerDynamic:
            if (!checkPrecision({Precision::FP32, Precision::FP32}, Precision::FP32)) {
                return false;
            }
            break;
        // TODO: debug: PRelu is not implemented
        //case Algorithm::EltwiseRelu:
        // TODO: debug: wip
        //case Algorithm::EltwiseExp:
            if (!checkPrecision({Precision::FP32}, Precision::FP32)) {
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

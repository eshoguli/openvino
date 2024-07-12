// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"
// arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "acl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"

namespace ov {
namespace intel_cpu {

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs, const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    fullyConnectedLayerInfo.transpose_weights = !attrs.weightsNonTransposed;
    if (!attrs.dequantizationScales.empty()) {
        dequantizationScale = attrs.dequantizationScales[0];
    }

    // Add postops
    if (!postOps.empty() && postOps.size() == 1) {
        if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
            fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                                             activation->alpha(),
                                                                             activation->beta(),
                                                                             activation->gamma());
        }
    }
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    // MatMul:
    // - check weights layout
    // FullyConnected:
    // - doesn't support dequantisation
    // - check weights layout

    // issue #<create and put number here>
//    const auto attrs = static_cast<FCAttrs>(config.attrs);
//    if (std::any_of(
//            attrs.dequantizationScales.begin(),
//            attrs.dequantizationScales.end(),
//            [](float value) { return value != 1.f;})) {
//        return false;
//    }

    const auto src1_dims = std::dynamic_pointer_cast<BlockedMemoryDesc>(config.descs.at(ARG_SRC))->getBlockDims();
    const auto src2_dims = std::dynamic_pointer_cast<BlockedMemoryDesc>(config.descs.at(ARG_WEI))->getBlockDims();

    VERIFY(one_of(srcType(config), ov::element::f16, ov::element::f32, ov::element::i8), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2,          UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    // TODO: FullyConnected only
    //VERIFY(one_of(weiRank(config), 2U, 3U),     UNSUPPORTED_WEI_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U, 4U),     UNSUPPORTED_WEI_RANK);
    VERIFY(static_cast<FCAttrs>(config.attrs).dequantizationScales.size() <= 1, UNSUPPORTED_PER_CHANNEL_QUANTIZATION);
    return true;
}

void ACLFullyConnectedExecutor::updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) {
    const auto src1_dims = aclMemoryShapes[ACLArgs::ACL_SRC_0];
    const auto src2_dims = aclMemoryShapes[ACLArgs::ACL_WEI];

    if (aclMemoryShapes[ACLArgs::ACL_WEI].num_dimensions() == 3U) {
        aclMemoryShapes[ACLArgs::ACL_WEI] = arm_compute::TensorShape(
                {aclMemoryShapes[ACLArgs::ACL_WEI][0] * aclMemoryShapes[ACLArgs::ACL_WEI][1],
                 aclMemoryShapes[ACLArgs::ACL_WEI][2]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_SRC_0] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][0],
            aclMemoryShapes[ACLArgs::ACL_SRC_0].total_size() / aclMemoryShapes[ACLArgs::ACL_WEI][0]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_DST].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][1],
            aclMemoryShapes[ACLArgs::ACL_SRC_0][1]});
    }

    // TODO: why we need it???
    if (!fullyConnectedLayerInfo.transpose_weights) {
        std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
    }
}

arm_compute::Status ACLFullyConnectedExecutor::validateTensorsInfo(const ACLMemoryInfo & aclMemoryInfos) {
    const auto fullyConnected = arm_compute::NEFullyConnectedLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);

    const auto& src1 = aclMemoryInfos[ACLArgs::ACL_SRC_0].get();
    const auto& shape1 = src1->tensor_shape();
    const auto& src2 = aclMemoryInfos[ACLArgs::ACL_WEI].get();
    const auto& shape2 = src2->tensor_shape();

    const auto matMulValid = arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            gemmInfo);
    return matMulValid;
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLMemoryTensors & aclMemoryTensors) {
//    const auto dstTensor = aclMemoryTensors.at(ACLArgs::ACL_DST).get();
//    if (dequantizationScale != 1.0) {
//        dstTensor->info()->set_quantization_info(arm_compute::QuantizationInfo(dequantizationScale, 0));
//    }
//
//    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
//    neFC->configure(
//            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
//            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
//            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
//            dstTensor,
//            fullyConnectedLayerInfo,
//            weightsInfo);
//    return neFC;

    auto matMull = std::make_unique<arm_compute::NEGEMMLowpMatrixMultiplyCore>();
    matMull->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
            aclMemoryTensors.at(ACLArgs::ACL_DST).get(),
            gemmInfo);
    return matMull;
}

ACLInfo ACLFullyConnectedExecutor::initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                                  const arm_compute::DataType& dataType,
                                                  const arm_compute::DataLayout& dataLayout) {
    arm_compute::DataType fcDataType;
    switch (dataType) {
        case arm_compute::DataType::S8: {
            fcDataType = arm_compute::DataType::QASYMM8_SIGNED;
            break;
        }
        case arm_compute::DataType::U8: {
            fcDataType = arm_compute::DataType::QASYMM8;
            break;
        }
        default: {
            fcDataType = dataType;
            break;
        }
    }

    return ACLCommonExecutor::initTensorInfo(tensorShape, fcDataType, dataLayout);
}

}   // namespace intel_cpu
}   // namespace ov

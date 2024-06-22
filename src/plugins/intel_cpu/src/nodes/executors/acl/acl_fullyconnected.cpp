// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"

#include <memory>

#include "acl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "acl_transpose.hpp"

namespace ov {
namespace intel_cpu {

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs, const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) : withBias(attrs.withBias) {
    if (!postOps.empty()) {
        // TODO: throw exception here
    }

    if (lowPrecision) {
        const auto transpose_executor = std::make_shared<ACLTransposeExecutor>(context);
        preOpExecutors.push_back(transpose_executor);

        gemmInfo.set_pretranspose_A(false);
        //gemmInfo.set_pretranspose_B(!attrs.weightsNonTransposed);
        gemmInfo.set_pretranspose_B(false);

//        const auto shape_a = aclMemoryInfoArgs.at(ARG_SRC).get()->tensor_shape();
//        const auto shape_b = aclMemoryInfoArgs.at(ARG_WEI).get()->tensor_shape();
    } else {
        aclTensorAttrs.enableNHWCReshape = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
        fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
        fullyConnectedLayerInfo.transpose_weights = !attrs.weightsNonTransposed;
        if (memory.at(ARG_SRC)->getPrecision() == ov::element::f16) {
            fullyConnectedLayerInfo.fp_mixed_precision = true;
        }

        // Add postops
        if (!postOps.empty() && postOps.size() == 1) {
            if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
                fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(
                        convertToEltwiseAlgorithm(activation->type()),
                        activation->alpha(),
                        activation->beta(),
                        activation->gamma());
            }
        }
    }
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    return true;

    if (true) {
        return true;
    } else {
        if (!config.postOps.empty() && config.postOps.size() != 1) {
            DEBUG_LOG("ACLFullyConnectedExecutor supports only 1 post op");
            return false;
        }

        const auto &srcDesc = config.descs.at(ARG_SRC);
        if (!one_of(srcDesc->getShape().getDims().size(), 2, 3, 4)) {
            DEBUG_LOG("ACLFullyConnectedExecutor supports only 2, 3 or 4 dimensions for inputs");
            return false;
        }

        const auto &weiDesc = config.descs.at(ARG_WEI);
        if (!one_of(weiDesc->getShape().getDims().size(), 2, 3)) {
            DEBUG_LOG("ACLFullyConnectedExecutor supports only 2 or 3 dimensions for weights");
            return false;
        }
        return true;
    }
}

void ACLFullyConnectedExecutor::prepareTensorsInfo() {
    if (lowPrecision) {
        const auto shape_a = aclMemoryInfoArgs.at(ARG_SRC).get()->tensor_shape();
        const auto shape_b = aclMemoryInfoArgs.at(ARG_WEI).get()->tensor_shape();

        tensorsInfoValidateStatus = arm_compute::Status(arm_compute::ErrorCode::OK);
    } else {
        auto wei_shape = aclMemoryInfoArgs.at(ARG_WEI)->tensor_shape();
        if (wei_shape.num_dimensions() == 3) {
            aclMemoryInfoArgs.at(ARG_WEI)->set_tensor_shape({wei_shape[0] * wei_shape[1], wei_shape[2]});
            wei_shape = aclMemoryInfoArgs.at(ARG_WEI)->tensor_shape();
        }

        auto src_shape = aclMemoryInfoArgs.at(ARG_SRC)->tensor_shape();
        if (one_of(src_shape.num_dimensions(), 3, 4)) {
            aclMemoryInfoArgs.at(ARG_SRC)->set_tensor_shape({wei_shape[0], src_shape.total_size() / wei_shape[0]});
            src_shape = aclMemoryInfoArgs.at(ARG_SRC)->tensor_shape();
        }

        if (one_of(aclMemoryInfoArgs.at(ARG_DST)->tensor_shape().num_dimensions(), 3, 4)) {
            aclMemoryInfoArgs.at(ARG_DST)->set_tensor_shape({wei_shape[1], src_shape[1]});
        }

        auto expected_weight_format = arm_compute::WeightFormat::UNSPECIFIED;
        weightsInfo = arm_compute::WeightsInfo(false, 1, 1,
                                               aclMemoryInfoArgs.at(ARG_WEI)->tensor_shape().total_size(),
                                               false, expected_weight_format);

        tensorsInfoValidateStatus = arm_compute::NEFullyConnectedLayer::has_opt_impl(
                expected_weight_format,
                aclMemoryInfoArgs.at(ARG_SRC).get(),
                aclMemoryInfoArgs.at(ARG_WEI).get(),
                withBias ? aclMemoryInfoArgs.at(ARG_BIAS).get() : nullptr,
                aclMemoryInfoArgs.at(ARG_DST).get(),
                fullyConnectedLayerInfo,
                weightsInfo);
        if (!tensorsInfoValidateStatus) { return; }
        fullyConnectedLayerInfo.enable_fast_math = arm_compute::is_fixed_format_fast_math(expected_weight_format);

        if (!fullyConnectedLayerInfo.transpose_weights) {
            arm_compute::TensorShape temp_weights_shape = aclMemoryInfoArgs.at(ARG_WEI)->tensor_shape();
            std::swap(temp_weights_shape[0], temp_weights_shape[1]);
            aclMemoryInfoArgs.at(ARG_WEI)->set_tensor_shape(temp_weights_shape);
        }
    }

    if (lowPrecision) {
        // why gemmInfo doesn't affect anything
        gemmInfo.set_pretranspose_A(false);
        gemmInfo.set_pretranspose_B(true);
        tensorsInfoValidateStatus = arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
                aclMemoryInfoArgs.at(ARG_SRC).get(),
                aclMemoryInfoArgs.at(ARG_WEI).get(),
                withBias ? aclMemoryInfoArgs.at(ARG_BIAS).get() : nullptr,
                aclMemoryInfoArgs.at(ARG_DST).get(),
                gemmInfo);

        if (tensorsInfoValidateStatus.error_code() != arm_compute::ErrorCode::OK) {
            const auto shape_a = aclMemoryInfoArgs.at(ARG_SRC).get()->tensor_shape();
            const auto shape_b = aclMemoryInfoArgs.at(ARG_WEI).get()->tensor_shape();
            std::cout << "error description: " << tensorsInfoValidateStatus.error_description() << std::endl;
        }
    } else {
        tensorsInfoValidateStatus = arm_compute::NEFullyConnectedLayer::validate(
                aclMemoryInfoArgs.at(ARG_SRC).get(),
                aclMemoryInfoArgs.at(ARG_WEI).get(),
                withBias ? aclMemoryInfoArgs.at(ARG_BIAS).get() : nullptr,
                aclMemoryInfoArgs.at(ARG_DST).get(),
                fullyConnectedLayerInfo,
                weightsInfo);
    }
}

void ACLFullyConnectedExecutor::configureFunction() {
    if (lowPrecision) {
        //ov::intel_cpu::TransposeParams &transposeParams
        //transpose_executor->init(attrs, postOps, memory, context);
        iFunction = std::make_unique<arm_compute::NEGEMMLowpMatrixMultiplyCore>();
    } else {
        iFunction = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    }

    // TODO: workaround: refactor: generalize
    // std::shared_ptr<arm_compute::Tensor>
    const auto& src_tensor = aclMemoryArgs.at(ARG_SRC);
    src_tensor->info()->set_quantization_info(arm_compute::QuantizationInfo(4.0f / 255.f, 0));

    const auto& weights_tensor = aclMemoryArgs.at(ARG_WEI);
    weights_tensor->info()->set_quantization_info(arm_compute::QuantizationInfo(8.0f / 255.f, 0));

    const auto& dst_tensor = aclMemoryArgs.at(ARG_DST);
    // without snippets: dequantization is fused but not used
    dst_tensor->info()->set_quantization_info(arm_compute::QuantizationInfo(32.f / 255.f, 0));
    // incorrect behaviour: with snippets
    //dst_tensor->info()->set_quantization_info(arm_compute::QuantizationInfo(1.f, 0));

    if (lowPrecision) {
        reinterpret_cast<arm_compute::NEGEMMLowpMatrixMultiplyCore *>(iFunction.get())->configure(
                aclMemoryArgs.at(ARG_SRC).get(),
                aclMemoryArgs.at(ARG_WEI).get(),
                withBias ? aclMemoryArgs.at(ARG_BIAS).get() : nullptr,
                aclMemoryArgs.at(ARG_DST).get(),
                gemmInfo);
    } else {
        reinterpret_cast<arm_compute::NEFullyConnectedLayer *>(iFunction.get())->configure(
                aclMemoryArgs.at(ARG_SRC).get(),
                aclMemoryArgs.at(ARG_WEI).get(),
                withBias ? aclMemoryArgs.at(ARG_BIAS).get() : nullptr,
                aclMemoryArgs.at(ARG_DST).get(),
                fullyConnectedLayerInfo,
                weightsInfo);
    }
}

// TODO: empty method: remove
bool ACLFullyConnectedExecutor::update(const MemoryArgs &memory) {
    return ACLCommonExecutor::update(memory);
}

}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

ACLMemoryInfo ACLCommonExecutor::initTensorInfo(
        const MemoryPtr& memoryPtr,
        const ACLTensorAttrs attrs,
        const QuantizedDataType quantized) {
    auto acl_tensor_type = precisionToAclDataType(memoryPtr->getPrecision(), quantized);
    auto acl_tensor_layout = getAclDataLayoutByMemoryDesc(memoryPtr->getDescPtr());

    ACLMemoryInfo aclMemoryInfo = nullptr;
    if (acl_tensor_type != arm_compute::DataType::UNKNOWN) {
        auto collapsed_dims = collapse_dims_to_max_rank(memoryPtr->getStaticDims(), attrs.maxDimsShape);
        auto acl_tensor_shape = shapeCast(collapsed_dims);
        if (attrs.hasLayoutTypeNHWC) {
            changeLayoutToNH_C({&acl_tensor_shape});
        }
        aclMemoryInfo = std::make_shared<arm_compute::TensorInfo>(
                acl_tensor_shape, 1,
                acl_tensor_type,
                acl_tensor_layout);
    }
    return aclMemoryInfo;
}

ACLMemory ACLCommonExecutor::initTensor(const ACLMemoryInfo& aclMemoryInfo) {
    ACLMemory aclMemory = nullptr;
    if (aclMemoryInfo) {
        aclMemory = std::make_shared<arm_compute::Tensor>();
        aclMemory->allocator()->init(*aclMemoryInfo);
    }
    return aclMemory;
}

bool ACLCommonExecutor::update(const MemoryArgs &memory) {
    for (auto& cpu_mem_ptr : memory) {
        auto aclPrecision = precisionToAclDataType(cpu_mem_ptr.second->getPrecision());
        QuantizedDataType quantized;
        switch (aclPrecision) {
            case arm_compute::DataType::S8: {
                quantized = QuantizedDataType::QASYMM;
                break;
            }
            case arm_compute::DataType::U8: {
                quantized = QuantizedDataType::QSYMM;
                break;
            }
            default: {
                quantized = QuantizedDataType::NONE;
                break;
            }
        }

        // Initialize arm_compute::TensorInfo object
        auto aclTensorInfo = initTensorInfo(cpu_mem_ptr.second, aclTensorAttrs, quantized);

        //TODO: debugging only
        if (aclTensorInfo->data_type() == arm_compute::DataType::F32) {
            aclTensorInfo->set_data_type(arm_compute::DataType::S32);
        }

        // Initialize arm_compute::Tensor object
        aclMemoryMap[cpu_mem_ptr.first] = initTensor(aclTensorInfo);
    }

    const auto src1 = aclMemoryMap.at(ARG_SRC)->info();
    const auto src2 = aclMemoryMap.at(ARG_WEI)->info();
    const auto src3 = aclMemoryMap.at(ARG_BIAS) ? aclMemoryMap.at(ARG_BIAS)->info() : nullptr;
    const auto dst = aclMemoryMap.at(ARG_DST)->info();

    // Update arm_compute::TensorInfo objects for specific ACL function
    auto tensorsInfoValidateStatus = updateTensorsInfo(aclMemoryMap);
    if (!tensorsInfoValidateStatus) {
        DEBUG_LOG("ACL operator validation was failed: ", tensorsInfoValidateStatus.error_description());
        return false;
    }

    // Configure arm_compute::IFunction object
    configureThreadSafe([&] {
        iFunction = configureFunction(aclMemoryMap);
    });
    return true;
}

void ACLCommonExecutor::execute(const MemoryArgs &memory) {
    for (auto& acl_tensor : aclMemoryMap) {
        if (acl_tensor.second) {
            acl_tensor.second->allocator()->import_memory(memory.at(acl_tensor.first)->getData());
        }
    }
    iFunction->run();
}

ACLCommonExecutor::~ACLCommonExecutor() {
    for (auto& acl_tensor : aclMemoryMap) {
        if (acl_tensor.second) {
            acl_tensor.second->allocator()->free();
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_executor.hpp"
#include "acl_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

bool ACLCommonExecutor::update(const MemoryArgs &memory) {
    std::unordered_map<int, arm_compute::DataType>   acl_tensors_types_list;
    std::unordered_map<int, arm_compute::DataLayout> acl_tensors_layouts_list;
    for (auto& cpu_mem_ptr : memory) {
        acl_tensors_types_list[cpu_mem_ptr.first] = precisionToAclDataType(cpu_mem_ptr.second->getPrecision());
        acl_tensors_layouts_list[cpu_mem_ptr.first] = getAclDataLayoutByMemoryDesc(cpu_mem_ptr.second->getDescPtr());
    }

    for (auto& cpu_mem_ptr : memory) {
        if (acl_tensors_types_list[cpu_mem_ptr.first] == arm_compute::DataType::UNKNOWN) {
            aclMemoryInfoArgs[cpu_mem_ptr.first] = std::make_shared<arm_compute::TensorInfo>();
            continue;
        }

        auto collapsed_dims = collapse_dims_to_max_rank(cpu_mem_ptr.second->getStaticDims(),
                                                        aclTensorAttrs.maxDimsShape);
        auto acl_tensor_shape = shapeCast(collapsed_dims);
        if (aclTensorAttrs.enableNHWCReshape) {
            changeLayoutToNH_C({&acl_tensor_shape});
        }
        aclMemoryInfoArgs[cpu_mem_ptr.first] = std::make_shared<arm_compute::TensorInfo>(
                acl_tensor_shape, 1,
                acl_tensors_types_list[cpu_mem_ptr.first],
                acl_tensors_layouts_list[cpu_mem_ptr.first]);
    }

    this->prepareTensorsInfo();
    if (!tensorsInfoValidateStatus) {
        DEBUG_LOG("ACL operator validation was failed: ", tensorsInfoValidateStatus.error_description());
        return false;
    }

    for (auto& acl_tensor_info : aclMemoryInfoArgs) {
        aclMemoryArgs[acl_tensor_info.first] = std::make_shared<arm_compute::Tensor>();
        aclMemoryArgs[acl_tensor_info.first]->allocator()->init(*acl_tensor_info.second);
    }

    configureThreadSafe([&] { this->configureFunction(); });
    return true;
}

void ACLCommonExecutor::execute(const MemoryArgs &memory) {
    for (auto& acl_tensor : aclMemoryArgs) {
        acl_tensor.second->allocator()->import_memory(memory.at(acl_tensor.first)->getData());
    }
    iFunction->run();
    for (auto& acl_tensor : aclMemoryArgs) {
        acl_tensor.second->allocator()->free();
    }
}

}   // namespace intel_cpu
}   // namespace ov

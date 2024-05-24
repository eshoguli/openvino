// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

using ACLMemoryArgs     = std::unordered_map<int, std::shared_ptr<arm_compute::Tensor>>;
using ACLMemoryInfoArgs = std::unordered_map<int, std::shared_ptr<arm_compute::TensorInfo>>;
using ACLFunction       = std::unique_ptr<arm_compute::IFunction>;

struct ACLTensorAttrs {
    bool enableNHWCReshape = false;
    size_t maxDimsShape = arm_compute::MAX_DIMS;
};

class ACLCommonExecutor : public Executor {
public:
    virtual void prepareTensorsInfo() {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'prepareTensorsInfo' method is not implemented by executor");
    }
    virtual void configureFunction() {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'configureFunction' method is not implemented by executor");
    }
    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }
    void execute(const MemoryArgs& memory) override;
    bool update(const MemoryArgs& memory) override;

protected:
    arm_compute::Status tensorsInfoValidateStatus;
    ACLFunction iFunction = nullptr;
    ACLMemoryArgs aclMemoryArgs;
    ACLMemoryInfoArgs aclMemoryInfoArgs;
    ACLTensorAttrs aclTensorAttrs;
};

using ACLCommonExecutorPtr = std::shared_ptr<ACLCommonExecutor>;

}  // namespace intel_cpu
}  // namespace ov

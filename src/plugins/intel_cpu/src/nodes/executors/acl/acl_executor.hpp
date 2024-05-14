// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

struct ACLTensorAttrs {
    bool enableNHWCReshape = false;
    size_t maxDimsShape = arm_compute::MAX_DIMS;
};

class ACLCommonExecutor : public Executor {
public:
    virtual arm_compute::Status prepare_tensors_info() = 0;
    virtual std::unique_ptr<arm_compute::IFunction> configure_function() = 0;

protected:
    std::unique_ptr<arm_compute::IFunction> ifunc = nullptr;
    std::unordered_map<int, arm_compute::Tensor> list_acl_tensors;
    std::unordered_map<int, arm_compute::TensorInfo> list_acl_tensors_infos;
    ACLTensorAttrs aclTensorAttrs;

private:
    void execute(const MemoryArgs& memory) override;
    bool update(const MemoryArgs& memory) override;
};

using ACLCommonExecutorPtr = std::shared_ptr<ACLCommonExecutor>;

}  // namespace intel_cpu
}  // namespace ov

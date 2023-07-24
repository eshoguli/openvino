// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_list.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#include "aarch64/jit_eltwise.hpp"
#endif

namespace ov {
namespace intel_cpu {

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList() {
    static std::vector<EltwiseExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclEltwiseExecutorBuilder>())
        // TODO: debug: commented to fix: getEltwiseExecutorsList() is used by ACL Executor only
        // OV_CPU_INSTANCE_ARCH_ARM64(ExecutorType::Aarch64, std::make_shared<ov::intel_cpu::executors::aarch64::JitEltwiseExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov

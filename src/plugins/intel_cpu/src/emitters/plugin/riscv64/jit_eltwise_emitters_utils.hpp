// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

// ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
//     std::vector<ov::element::Type> input_precisions;
//     for (const auto& input : n->inputs()) {
//         input_precisions.push_back(
//             input.get_source_output().get_element_type());
//     }

//     assert(std::all_of(
//         input_precisions.begin(),
//         input_precisions.end(),
//         [&input_precisions](const ov::element::Type& precision) {return precision == input_precisions[0]; }));

//     return input_precisions[0];
// }

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

class PropagatePrecision: ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("PropagatePrecision", "0");
    PropagatePrecision(
        const ov::element::Type supported_precision,
        const std::shared_ptr<const TargetMachine>& target_machine);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const ov::element::Type supported_precision;
    const std::shared_ptr<const TargetMachine> target_machine;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

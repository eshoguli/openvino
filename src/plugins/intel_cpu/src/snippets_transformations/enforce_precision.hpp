// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

class EnforcePrecision: public ngraph::pass::FunctionPass {
public:
    class Operation {
    public:
        bool only_after_parameter;
        bool require_target_isa;
        std::set<std::vector<element::Type>> precisions;
    };

    OPENVINO_RTTI("EnforcePrecision", "0");

    EnforcePrecision(
        const element::Type source,
        const element::Type target,
        const bool is_target_isa_supported,
        std::function<EnforcePrecision::Operation(const std::shared_ptr<ngraph::Node>& op)> get_supported_precisions = nullptr);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    static EnforcePrecision::Operation get_supported_precisions_default(const std::shared_ptr<ngraph::Node>& op) noexcept;

    const element::Type source;
    const element::Type target;
    const bool is_target_isa_supported;
    const std::function<EnforcePrecision::Operation(const std::shared_ptr<ngraph::Node>& op)> get_supported_precisions;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov

// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>
#include <snippets/itt.hpp>
#include "snippets/op/convert_saturation.hpp"

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

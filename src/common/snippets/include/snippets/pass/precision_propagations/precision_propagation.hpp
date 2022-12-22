// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagation {

class PrecisionPropagation: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("PrecisionPropagation", "0");
    PrecisionPropagation(const ov::element::Type exec_type = ov::element::f32);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    ov::element::Type exec_type;
};

}  // namespace precision_propagation
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

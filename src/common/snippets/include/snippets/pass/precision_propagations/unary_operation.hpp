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

class UnaryOperation: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("UnaryOperation", "0");
    UnaryOperation(const ov::element::Type exec_type = ov::element::f32);
private:
    ov::element::Type exec_type;
};

}  // namespace precision_propagation
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

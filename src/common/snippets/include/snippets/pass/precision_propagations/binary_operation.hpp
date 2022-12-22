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

class BinaryOperation: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BinaryOperation", "0");
    BinaryOperation(const std::vector<std::pair<ov::element::Type, ov::element::Type>>& precisions = {});
private:
    const std::vector<std::pair<ov::element::Type, ov::element::Type>> precisions;
};

}  // namespace precision_propagation
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

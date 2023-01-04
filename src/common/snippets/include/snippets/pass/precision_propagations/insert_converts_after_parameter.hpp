// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagations {

class InsertConvertsAfterParameters : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertConvertsAfterParameters", "0");
    InsertConvertsAfterParameters(const ov::element::Type supported_precision);
};

}  // precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

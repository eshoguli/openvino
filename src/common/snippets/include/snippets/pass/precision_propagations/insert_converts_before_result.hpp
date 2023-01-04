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

class InsertConvertsBeforeResults: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertConvertsBeforeResults", "0");
    InsertConvertsBeforeResults(const ov::element::Type supported_precision);
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

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

class InsertConverts: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertConverts", "0");
    InsertConverts(const ov::element::Type supported_precision);
};

}  // namespace precision_propagation
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

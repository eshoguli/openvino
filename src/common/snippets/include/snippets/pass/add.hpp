// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

class Add: public ngraph::pass::MatcherPass {
public:
    Add();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

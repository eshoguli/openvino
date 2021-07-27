// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "ngraph/pass/graph_rewrite.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ParallelGraphRewrite;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class LP_TRANSFORMATIONS_API ngraph::pass::low_precision::ParallelGraphRewrite : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    bool apply_matcher_passes(std::shared_ptr<Function> f, std::deque<std::shared_ptr<Node>> nodes_to_run);
};


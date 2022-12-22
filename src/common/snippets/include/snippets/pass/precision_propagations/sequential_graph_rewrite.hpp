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

class SequentialGraphRewrite : ov::pass::GraphRewrite {
public:
	explicit SequentialGraphRewrite(const std::shared_ptr<ov::pass::MatcherPass>& pass) : GraphRewrite(pass) {}

	bool apply_matcher_pass(std::shared_ptr<ov::Model> f, std::weak_ptr<Node> node_to_run);
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

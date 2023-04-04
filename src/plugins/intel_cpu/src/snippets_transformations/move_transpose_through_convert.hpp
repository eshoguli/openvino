// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

// TODO: add TODO that we should move not always: precision increasing
class MoveTransposeThroughConvert : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MoveTransposeThroughConvert", "0");

    MoveTransposeThroughConvert(
        const element::Type source,
        const element::Type target);

private:
    const element::Type source;
    const element::Type target;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov

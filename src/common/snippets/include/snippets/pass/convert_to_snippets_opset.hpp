// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>
#include <snippets/itt.hpp>
#include "snippets/op/convert_saturation.hpp"

#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

class ConvertToSnippetsOpset : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertToSnippetsOpset", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/layer_transformation.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API AlignQuantizationIntervals;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::AlignQuantizationIntervals : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AlignQuantizationIntervals(LayerTransformation::Params params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    LayerTransformation::Params params;
};

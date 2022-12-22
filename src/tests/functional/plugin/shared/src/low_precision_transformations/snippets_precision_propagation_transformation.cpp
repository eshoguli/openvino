// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/snippets_precision_propagation_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/snippets_precision_propagation_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string SnippetsPrecisionPropagationTransformation::getTestCaseName(const testing::TestParamInfo<SnippetsPrecisionPropagationTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    SnippetsPrecisionPropagationTransformationValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        param.inputShape1 << "_" << param.precision1 << "_" <<
        param.inputShape2 << " " << param.precision2;
    return result.str();
}

void SnippetsPrecisionPropagationTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    SnippetsPrecisionPropagationTransformationValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::SnippetsPrecisionPropagationFunction::get(
        param.precision1,
        param.inputShape1,
        param.precision2,
        param.inputShape2);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
}

TEST_P(SnippetsPrecisionPropagationTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class SnippetsPrecisionPropagationTransformationValues{
public:
    ngraph::element::Type precision1;
    ngraph::PartialShape inputShape1;
    ngraph::element::Type precision2;
    ngraph::PartialShape inputShape2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    SnippetsPrecisionPropagationTransformationValues
> SnippetsPrecisionPropagationTransformationParams;

class SnippetsPrecisionPropagationTransformation :
    public testing::WithParamInterface<SnippetsPrecisionPropagationTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SnippetsPrecisionPropagationTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

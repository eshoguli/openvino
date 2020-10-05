// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/low_precision_transformations/fuse_convert_function.hpp"

#include <ngraph/pass/visualize_tree.hpp>

namespace LayerTestsDefinitions {

std::string FuseConvertTransformation::getTestCaseName(testing::TestParamInfo<FuseConvertTransformationParams> obj) {
    std::string targetDevice;
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, version, deqOperations, constInput) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params, version) <<
           "_" << deqOperations << "_" << constInput;
    return result.str();
}

void FuseConvertTransformation::SetUp() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, version, deqOperations, constInput) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::FuseConvertFunction::getWithFQ(
        shape,
        precision,
        deqOperations,
        constInput);
}

TEST_P(FuseConvertTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

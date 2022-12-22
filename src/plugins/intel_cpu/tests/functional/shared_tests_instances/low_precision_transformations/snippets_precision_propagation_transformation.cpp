// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/snippets_precision_propagation_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<LayerTestsDefinitions::SnippetsPrecisionPropagationTransformationValues> params = {
    //{
    //    ngraph::element::f32,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //    ngraph::element::f32,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //},
    //{
    //    ngraph::element::u8,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //    ngraph::element::u8,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //},
    {
        ngraph::element::i8,
        ngraph::PartialShape({1, 3, 16, 16}),
        ngraph::element::i8,
        ngraph::PartialShape({1, 3, 16, 16}),
    },
    //{
    //    ngraph::element::i32,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //    ngraph::element::i32,
    //    ngraph::PartialShape({ 1, 3, 16, 16 }),
    //}
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, SnippetsPrecisionPropagationTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(params)),
    SnippetsPrecisionPropagationTransformation::getTestCaseName);
}  // namespace

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<MatMulShapes> shapes = {
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 16, 8 },
        false,
        false
    },
    {
        ov::PartialShape{ 1, 1, 16 },
        ov::PartialShape{ 1, 16, 8 },
        false,
        false
    },
//    // transposeB
//    {
//        ov::PartialShape{ 1, 16 },
//        ov::PartialShape{ 8, 16 },
//        false,
//        true
//    },
    {
        ov::PartialShape{ 16, 1 },
        ov::PartialShape{ 16, 8 },
        true,
        false
    },
    {
        ov::PartialShape{ 1, 16, 1 },
        ov::PartialShape{ 1, 16, 8 },
        true,
        false
    },
//    // MatMul_101
//    {
//        ov::PartialShape{ 1, 128, 768 },
//        ov::PartialShape{ 3072, 768 },    // after transpose: 768 x 3072
//        false,
//        true
//    },
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<FullyConnectedParams> activations = {
    // TODO: failed if transposeB = true: accuracy check
    {
        true,  // activation
        false, // perChannel
        "fullyConnected,fullyConnected/DequantizationMultiply,relu"
    },
//    // TODO: failed if transposeB = true: fp32 execution, FQ is not decomposed
//    {
//        true,  // activation
//        true,  // perChannel
//        "fullyConnected,fullyConnected/DequantizationMultiply,relu"
//    },
    {
        false,  // activation
        false,  // perChannel
        "fullyConnected_original,fullyConnected"
    },
//    // TODO: failed if transposeB = true: fp32 execution, FQ is not decomposed
//    {
//        false, // activation
//        true,  // perChannel
//        "fullyConnected_original,fullyConnected"
//    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn({ov::element::i8 /*, ov::element::u8*/}),
        ::testing::ValuesIn(activations),
        ::testing::Values("gemm_acl_i8")),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace

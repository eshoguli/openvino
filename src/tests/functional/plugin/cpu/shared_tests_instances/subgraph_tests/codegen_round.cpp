
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_round.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(CodeGeneration, CodegenRound,
    ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::SizeVector({1, 3, 16, 16})),
    ::testing::ValuesIn({ngraph::opset5::Round::RoundMode::HALF_TO_EVEN, ngraph::opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO}),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    CodegenRound::getTestCaseName);
}  // namespace

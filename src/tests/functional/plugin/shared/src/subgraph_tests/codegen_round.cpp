
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "subgraph_tests/codegen_round.hpp"
#include "ngraph/opsets/opset5.hpp"

namespace LayerTestsDefinitions {

std::string CodegenRound::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    ngraph::opset5::Round::RoundMode mode;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, mode, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "PRC=" << netPrecision.name() << "_";
    result << "M=" << (mode == ngraph::opset5::Round::RoundMode::HALF_TO_EVEN ? "HALF_TO_EVEN" : "HALF_AWAY_FROM_ZERO") << "_";
    result << "D=" << targetDevice;
    return result.str();
}

void CodegenRound::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        float* rawBlobDataPtr = blob->buffer().as<float*>();
        const size_t blobSize = blob->size();
        float value = 0.f;
        for (size_t i = 0; i < blobSize; i++) {
            value += 0.1f;
            rawBlobDataPtr[i] = value;
        }
        inputs.push_back(blob);
    }
}

void CodegenRound::SetUp() {
    std::vector<size_t> inputShape0;
    InferenceEngine::Precision netPrecision;
    ngraph::opset5::Round::RoundMode mode;
    std::tie(netPrecision, inputShape0, mode, targetDevice) = this->GetParam();

    auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape0});
    auto round = std::make_shared<ngraph::opset5::Round>(input, mode);
    auto result = std::make_shared<ngraph::opset1::Result>(round);

    function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result},
        ngraph::ParameterVector{input},
        "CodegenRound");
}

TEST_P(CodegenRound, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

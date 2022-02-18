
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "subgraph_tests/codegen_convert.hpp"

namespace LayerTestsDefinitions {

std::string CodegenConvert::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::inputParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    ov::element::Type inputPrecision;
    ov::element::Type outputPrecision;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, inputPrecision, outputPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << inputPrecision << "_";
    result << "OUT=" << outputPrecision;
    return result.str();
}

void CodegenConvert::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const bool is_signed = param->output(0).get_element_type().is_signed();

        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
        char* rawBlobDataPtr = blob->buffer().as<char*>();
        const size_t blobSize = blob->size();
        for (size_t i = 0; i < blobSize; i++) {
            rawBlobDataPtr[i] = static_cast<char>(is_signed ? (i - 128) : i);
        }
        inputs.push_back(blob);
    }
}

void CodegenConvert::SetUp() {
    std::vector<size_t> inputShape0;
    InferenceEngine::Precision netPrecision;
    ov::element::Type inputPrecision;
    ov::element::Type outputPrecision;
    std::tie(netPrecision, inputShape0, inputPrecision, outputPrecision, targetDevice) = this->GetParam();

    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape{inputShape0});
    const auto convert = std::make_shared<ngraph::opset1::Convert>(input, outputPrecision);
    const auto floor = std::make_shared<ngraph::opset1::Floor>(convert);
    const auto result = std::make_shared<ngraph::opset1::Result>(floor);

    function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{result},
        ngraph::ParameterVector{input},
        "CodegenConvert");
}

TEST_P(CodegenConvert, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

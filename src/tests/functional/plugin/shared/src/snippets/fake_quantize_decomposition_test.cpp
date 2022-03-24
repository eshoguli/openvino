
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fake_quantize_decomposition_test.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "fake_quantize_function.hpp"
#include "function_helper.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeDecompositionTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
    std::ostringstream result;
    const auto values = std::get<0>(obj.param);
    const auto operation = std::get<1>(obj.param);
    const auto operations_number = std::get<2>(obj.param);
    const auto targetDevice = std::get<3>(obj.param);

    const auto operationString = ngraph::is_type<ngraph::opset1::Parameter>(operation.first) ? "nullptr" : std::string(operation.first->get_type_name());

    result << "IS=" << CommonTestUtils::vec2str(values.inputShape) << "_";
    result << "netPRC=" << values.modelType << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << values.inputType << "_";
    result << "OP=" << operationString << "_";
    result << "ON1=" << std::string(operation.second.first) << "_";
    result << "ON1=" << std::string(operation.second.second) << "_";
    result << "LP=" << values.zeroPoint;
    result << "SH1=" << values.fakeQuantizeShapes[0] << "SH2=" << values.fakeQuantizeShapes[1]
           << "SH3=" << values.fakeQuantizeShapes[2] << "SH4=" << values.fakeQuantizeShapes[3];
    return result.str();
}

//void FakeQuantizeDecompositionTest::GenerateInputs() {
//    inputs.clear();
//    const auto& inputsInfo = executableNetwork.GetInputsInfo();
//    const auto& functionParams = function->get_parameters();
//    for (int i = 0; i < functionParams.size(); ++i) {
//        const auto& param = functionParams[i];
//        const auto infoIt = inputsInfo.find(param->get_friendly_name());
//        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
//        InferenceEngine::InputInfo::CPtr info = infoIt->second;
//        InferenceEngine::Blob::Ptr blob = GenerateInput(*info);
//        float* rawBlobDataPtr = blob->buffer().as<float*>();
//        const size_t blobSize = blob->size();
//        float value = 0;
//        for (size_t i = 0; i < blobSize; i++) {
//            rawBlobDataPtr[i] = value;
//            value = value + 1.f;
//            if (value > 255.f) {
//                value = 0.f;
//            }
//        }
//        inputs.push_back(blob);
//    }
//}

void FakeQuantizeDecompositionTest::SetUp() {
    auto& testsParams = this->GetParam();

    const auto values = std::get<0>(testsParams);
    const auto operation = std::get<1>(testsParams);
    const auto operations_number = std::get<2>(testsParams);
    targetDevice = std::get<3>(testsParams);

    //targetDevice = values.actual.targetDevice;
    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    init_input_shapes({{values.inputShape, {values.inputShape}}});

    std::shared_ptr<ngraph::Node> op = ngraph::is_type<ngraph::opset1::Parameter>(operation.first) ? nullptr : operation.first;
    function = ov::test::snippets::FakeQuantizeFunction::getOperationAndFakeQuantize(
        {values.inputShape},
        values.inputType,
        values.fakeQuantizeShapes,
        values.zeroPoint,
        ov::test::snippets::FunctionHelper::makePrerequisitesOriginal(),
        op);
}

TEST_P(FakeQuantizeDecompositionTest, CompareWithRefImpl) {
    run();

    const auto operation = std::get<1>(this->GetParam());
    auto elementType = std::string(operation.second.first);
    validateOriginalLayersNamesByType(elementType, operation.second.second);

    validateNumSubgraphs();
};

}  // namespace LayerTestsDefinitions

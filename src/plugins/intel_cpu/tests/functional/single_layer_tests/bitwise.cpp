// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include <shared_test_classes/single_layer/bitwise.hpp>
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
    std::pair<std::vector<size_t>, std::vector<size_t>>,    // Input shapes tuple
    ngraph::helpers::LogicalTypes,      // Logical op type
    ngraph::helpers::InputLayerType,    // Second input type
    InferenceEngine::Precision,         // Net precision
    InferenceEngine::Precision,         // Input precision
    InferenceEngine::Precision,         // Output precision
    InferenceEngine::Layout,            // Input layout
    InferenceEngine::Layout,            // Output layout
    std::string,                        // Device name
    std::map<std::string, std::string>  // Additional network configuration
> BitwiseTestParams;

typedef std::tuple<
        BitwiseTestParams,
        CPUSpecificParams>
BitwiseLayerCPUTestParamSet;

class BitwiseLayerCPUTest : public testing::WithParamInterface<BitwiseLayerCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BitwiseLayerCPUTestParamSet> obj) {
        BitwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        //result << LayerTestsDefinitions::BitwiseLayerTest::getTestCaseName(testing::TestParamInfo<BitwiseTestParams>(
        //        basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        BitwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::pair<std::vector<size_t>, std::vector<size_t>> inputShapes;
        ngraph::helpers::LogicalTypes logicalOpType;
        ngraph::helpers::InputLayerType secondInputType;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(inputShapes, logicalOpType, secondInputType, netPrecision, inPrc, outPrc,
                 inLayout, outLayout, targetDevice, additional_config) = basicParamsSet;

        selectedType = getPrimitiveType() + "_" + inPrc.name();

        auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(Precision::BOOL); // Because ngraph supports only boolean input for logical ops
        configuration.insert(additional_config.begin(), additional_config.end());

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first))};
        std::shared_ptr<ngraph::Node> logicalNode;
        if (logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT) {
            auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
            if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondInput));
            }
            logicalNode = ngraph::builder::makeLogical(inputs[0], secondInput, logicalOpType);
        } else {
            logicalNode = ngraph::builder::makeLogical(inputs[0], ngraph::Output<ngraph::Node>(), logicalOpType);
        }

        logicalNode->get_rt_info() = getCPUInfo();

        function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
    }
};

TEST_P(BitwiseLayerCPUTest, CompareWithRefs) {
    Run();
    CheckPluginRelatedResults(executableNetwork, "Eltwise");
}

namespace {

//std::map<std::vector<size_t>, std::vector<std::vector<size_t >>> inputShapes = {
//        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
//        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
//        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
//        {{1, 3, 20}, {{20}, {2, 1, 1}}},
//        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
//        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
//};
//
//std::map<std::vector<size_t>, std::vector<std::vector<size_t >>> inputShapesNot = {
//        {{1}, {}},
//        {{5}, {}},
//        {{2, 200}, {}},
//        {{1, 3, 20}, {}},
//        {{2, 17, 3, 4}, {}},
//        {{2, 1, 1, 3, 1}, {}},
//};

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inputShapes = {};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::BOOL,
};

std::vector<ngraph::helpers::LogicalTypes> logicalOpTypes = {
        ngraph::helpers::LogicalTypes::LOGICAL_AND,
        ngraph::helpers::LogicalTypes::LOGICAL_OR,
        ngraph::helpers::LogicalTypes::LOGICAL_XOR,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config;

std::vector<Precision> inpOutPrc = {Precision::FP32};

const auto bitwiseTestParams = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inputShapes),
            ::testing::ValuesIn(logicalOpTypes),
            ::testing::ValuesIn(secondInputTypes),
            ::testing::Values(Precision::BF16),
            ::testing::ValuesIn(inpOutPrc),
            ::testing::ValuesIn(inpOutPrc),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_Bitwise_Eltwise_CPU, BitwiseLayerCPUTest, bitwiseTestParams, BitwiseLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/pad.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
    ov::test::InputShape, // input_shapes
    std::vector<std::vector<size_t>>, // indices
    bool                 // with_weights
    > embeddingBagPackedSumParams;

typedef std::tuple<
    embeddingBagPackedSumParams,
    ov::test::ElementType, // embedding table
    ov::test::ElementType, // indices
    LayerTestsUtils::TargetDevice> embeddingBagPackedSumLayerTestParamsSet;

class EmbeddingBagPackedSumCPUTest : public testing::WithParamInterface<embeddingBagPackedSumLayerTestParamsSet>,
                        virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet>& obj) {
        embeddingBagPackedSumParams params;
        ov::test::ElementType netPrecision, indPrecision;
        std::string targetDevice;
        std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

        ov::test::InputShape inputShapes;
        std::vector<std::vector<size_t>> indices;
        bool withWeights;
        std::tie(inputShapes, indices, withWeights) = params;

        std::ostringstream result;
        result << "IS=" << inputShapes << "_";
        result << "I" << CommonTestUtils::vec2str(indices) << "_";
        result << "WW" << withWeights << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "indPRC=" << indPrecision << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        embeddingBagPackedSumParams embParams;
        ov::test::ElementType netPrecision, indPrecision;
        std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();

        ov::test::InputShape inputShapes;
        std::vector<std::vector<size_t>> indices;
        bool withWeights;
        std::tie(inputShapes, indices, withWeights) = embParams;

        init_input_shapes({ inputShapes });

        auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputShapes.first);
        ngraph::ParameterVector params = {emb_table_node};

        auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagPackedSum>(
                ngraph::builder::makeEmbeddingBagPackedSum(
                        netPrecision, indPrecision, emb_table_node, indices, withWeights));
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
        function = std::make_shared<ngraph::Function>(results, params, "embeddingBagPackedSum");
    }
};

TEST_P(EmbeddingBagPackedSumCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(executableNetwork, "EmbeddingBagPackedSum");
}

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
        ov::test::ElementType::f32,
        ov::test::ElementType::i32,
        ov::test::ElementType::u8
};

const std::vector<ov::test::ElementType> indPrecisions = {
        ov::test::ElementType::i64,
        ov::test::ElementType::i32
};

const std::vector<ov::test::InputShape> input_shapes = {
    // dynamic input shapes
    {
        // input model dynamic shapes
        {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        // input tensor shapes
        {{{5, 6}}, {10, 35}}
    },
    {
        // input model dynamic shapes
        {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        // input tensor shapes
        {{5, 4, 16}, {10, 12, 8}}
    },
    {
        // input model dynamic shapes with limits
        {{5, 10}, {6, 35}, {4, 8}},
        // input tensor shapes
        {{5, 6, 4}, {10, 35, 8}, {5, 6, 4}}
    },
    // static shapes
    {{5, 6}, {{5, 6}}},
    {{10, 35}, {{10, 35}}},
    {{5, 4, 16}, {{5, 4, 16}}},
};

const std::vector<std::vector<std::vector<size_t>>> indices =
        {{{0, 1}, {2, 2}, {3, 4}}, {{4, 4, 3}, {1, 0, 2}}, {{1, 2, 1, 2}, {1, 2, 1, 2}}};
const std::vector<bool> with_weights = {false, true};

const auto embBagPackedSumArgSet = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(with_weights)
);

INSTANTIATE_TEST_SUITE_P(smoke, EmbeddingBagPackedSumCPUTest,
     ::testing::Combine(
             embBagPackedSumArgSet,
             ::testing::ValuesIn(netPrecisions),
             ::testing::ValuesIn(indPrecisions),
             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
             EmbeddingBagPackedSumCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions


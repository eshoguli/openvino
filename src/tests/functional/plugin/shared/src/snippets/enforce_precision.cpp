// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/enforce_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "subgraph_transpose_matmul.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string EnforcePrecisionTest::getTestCaseName(testing::TestParamInfo<EnforcePrecisionTestParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    EnforcePrecisionTestValues test_values;
    size_t num_nodes, num_subgraphs;
    std::string targetDevice;
    std::tie(input_shapes, test_values, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    result << "transpose=" << test_values.transpose << "_";
    result << "mat_mul=" << test_values.mat_mul << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EnforcePrecisionTest::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    EnforcePrecisionTestValues test_values;
    std::tie(input_shapes, test_values, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    function = SubgraphTransposeMatMulFunction(
        test_values.mat_mul ? input_shapes : std::vector<ov::PartialShape>{input_shapes[0]},
        ov::element::f32,
        test_values.transpose,
        test_values.mat_mul).getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({ InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK });
    }

    setInferenceType(element::bf16);
}

TEST_P(EnforcePrecisionTest, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov

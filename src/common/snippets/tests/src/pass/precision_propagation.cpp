// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/precision_propagation.hpp"

#include <gtest/gtest.h>
#include <common_test_utils/common_utils.hpp>
#include <snippets/pass/propagate_precision.hpp>
#include "precision_propagation_function.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string PrecisionPropagationTest::getTestCaseName(testing::TestParamInfo<insertLoadStoreParams> obj) {
    std::vector<Shape> inputShapes(3);
    std::vector<Shape> broadcastShapes(3);
    std::tie(inputShapes[0], inputShapes[1], inputShapes[2],
             broadcastShapes[0], broadcastShapes[1], broadcastShapes[2]) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    for (size_t i = 0; i < broadcastShapes.size(); i++)
        result << "BS[" << i << "]=" << CommonTestUtils::vec2str(broadcastShapes[i]) << "_";
    return result.str();
}

void PrecisionPropagationTest::SetUp() {
    //TransformationTestsF::SetUp();
    //std::vector<Shape> inputShapes(3);
    //std::vector<Shape> broadcastShapes(3);
    //std::tie(inputShapes[0], inputShapes[1], inputShapes[2],
    //         broadcastShapes[0], broadcastShapes[1], broadcastShapes[2]) = this->GetParam();
    //function = PrecisionPropagationFunction::get(inputShapes, broadcastShapes);

    //snippets::pass::PropagatePrecision(element::f32, m_generator->get_target_machine()).run_on_model(function);

    //function_ref = PrecisionPropagationFunction::get(inputShapes, broadcastShapes);
}

TEST_P(PrecisionPropagationTest, CompareFunctions) {
    auto res = compare_functions(function, function_ref, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

namespace PrecisionPropagationTestInstantiation {
using ov::Shape;
std::vector<Shape> inputShapes{{1, 4, 1, 5, 1}, {1, 4, 2, 5, 1}};
std::vector<Shape> broadcastShapes{{1, 4, 1, 5, 16}, {1, 4, 2, 5, 16}};
Shape exec_domain{1, 4, 2, 5, 16};
Shape emptyShape{};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_PrecisionPropagationTest, PrecisionPropagationTest,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(inputShapes[0]),
                                 ::testing::Values(inputShapes[1]),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(broadcastShapes[0]),
                                 ::testing::Values(broadcastShapes[1])),
                         PrecisionPropagationTest::getTestCaseName);

} // namespace PrecisionPropagationTestInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov
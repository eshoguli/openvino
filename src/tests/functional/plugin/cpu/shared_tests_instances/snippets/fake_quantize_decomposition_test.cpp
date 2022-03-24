
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/fake_quantize_decomposition_test.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph;

namespace {

namespace decompositionIgnore {
const std::vector<TestValues> testValuesDecomposition = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<ngraph::opset1::Parameter>(), {"FakeQuantize", "fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecomposition),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{5, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
} // decompositionIgnore


namespace decompositionInSubgraph {
const std::vector<TestValues> testValuesDecomposition = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}},
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string> >> operations = {
    {std::make_shared<opset1::Abs>(), {"Subgraph", "relu2,Abs,fakeQuantize"}},
    {std::make_shared<opset1::Clamp>(), {"Subgraph", "relu2,Clamp,fakeQuantize"}},
    {std::make_shared<opset1::Floor>(), {"Subgraph", "relu2,Floor,fakeQuantize"}},
    {std::make_shared<opset1::Ceiling>(), {"Subgraph", "relu2,Ceiling,fakeQuantize"}},
    {std::make_shared<opset1::Elu>(), {"Subgraph", "relu2,Elu,fakeQuantize"}},
    {std::make_shared<opset1::Erf>(), {"Subgraph", "relu2,Erf,fakeQuantize"}},
    {std::make_shared<opset1::Exp>(), {"Subgraph", "relu2,Exp,fakeQuantize"}},
    {std::make_shared<opset1::LogicalNot>(), {"Subgraph", "relu2,LogicalNot,fakeQuantize"}},
    {std::make_shared<opset1::Negative>(), {"Subgraph", "relu2,Negative,fakeQuantize"}},
    {std::make_shared<opset1::Relu>(), {"Subgraph", "relu2,fakeQuantize"}},
    {std::make_shared<opset5::Round>(), {"Subgraph", "relu2,Round,fakeQuantize"}},
    {std::make_shared<opset1::Sigmoid>(), {"Subgraph", "relu2,Sigmoid,fakeQuantize"}},
    {std::make_shared<opset1::Sqrt>(), {"Subgraph", "relu2,Sqrt,fakeQuantize"}},
    {std::make_shared<opset1::Tanh>(), {"Subgraph", "relu2,Tanh,fakeQuantize"}},
    {std::make_shared<ngraph::op::v0::Gelu>(), {"Subgraph", "relu2,Gelu,fakeQuantize"}},
    //{std::make_shared<ngraph::op::v7::Gelu>(), {"Subgraph", "relu2,Gelu,fakeQuantize"}},
    {std::make_shared<ngraph::op::v4::HSwish>(), {"Subgraph", "relu2,HSwish,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecomposition),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{4, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
} // decompositionInSubgraph


namespace legacyFuse {
const std::vector<TestValues> testValuesLegacyFuse = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<opset1::Convolution>(), {"Convolution", "Convolution,fakeQuantize"}},
    //{std::make_shared<opset1::GroupConvolution>(), {"GroupConvolution", "GroupConvolution,fakeQuantize"}},
    //{std::make_shared<opset1::MatMul>(), {"MatMul", "MatMul,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesLegacyFuse),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{8, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

} // legacyFuse

}  // namespace

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_transpose_matmul.hpp"
#include <common_test_utils/data_utils.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ov {
namespace test {
namespace snippets {

SubgraphTransposeMatMulFunction::SubgraphTransposeMatMulFunction(
    const std::vector<ov::PartialShape>& input_shapes,
    const element::Type input_type,
    const bool transpose,
    const bool mat_mul) :
    SnippetsFunctionBase(input_shapes, input_type),
    transpose(transpose),
    mat_mul(mat_mul) {
}

std::shared_ptr<ov::Model> SubgraphTransposeMatMulFunction::get(
    const std::vector<ov::PartialShape>& input_shapes,
    const element::Type input_type,
    const bool transpose,
    const bool mat_mul) {
    const auto parameter1 = std::make_shared<op::v0::Parameter>(input_type, input_shapes[0]);
    parameter1->set_friendly_name("parameter1");

    const auto shift = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{ 1 }, std::vector<float>{1});
    const auto axes = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{ 1 }, std::vector<float>{0});

    std::shared_ptr<Node> parent1 = std::make_shared<ov::op::v7::Roll>(parameter1, shift, axes);
    parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    parent1->set_friendly_name("roll1");

    if (transpose) {
        parent1 = std::make_shared<op::v1::Transpose>(
            parent1,
            std::make_shared<op::v0::Constant>(element::i64, Shape{ input_shapes[0].size() }, std::vector<int>{0, 1, 3, 2}));
        parent1->set_friendly_name("transpose1");
        // TODO: should be removed
        parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;

        parent1 = std::make_shared<op::v1::Multiply>(
            parent1,
            std::make_shared<op::v0::Constant>(element::f32, Shape{}, std::vector<float>{2.0f}));
        parent1->set_friendly_name("multiply");
        parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;

        parent1 = std::make_shared<op::v1::Add>(
            parent1,
            std::make_shared<op::v0::Constant>(element::f32, Shape{}, std::vector<float>{2.0f}));
        parent1->set_friendly_name("add");
        parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;

        parent1 = std::make_shared<op::v1::Transpose>(
            parent1,
            std::make_shared<op::v0::Constant>(element::i64, Shape{ input_shapes[0].size() }, std::vector<int>{0, 1, 3, 2}));
        parent1->set_friendly_name("transpose2");
        // TODO: should be removed
        parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    }

    std::shared_ptr<ngraph::opset1::Parameter> parameter2;
    std::shared_ptr<Node> parent2;

    if (mat_mul) {
        parameter2 = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shapes[1]);
        parameter2->set_friendly_name("parameter2");

        parent2 = std::make_shared<ov::op::v7::Roll>(parameter2, shift, axes);
        parent2->get_rt_info()["enforceBF16evenForGraphTail"] = true;
        parent2->set_friendly_name("roll2");

        parent1 = std::make_shared<op::v0::MatMul>(parent1, parent2);
        parent1->set_friendly_name("matmul");
        // TODO: should be removed
        parent1->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    }

    auto roll3 = std::make_shared<ov::op::v7::Roll>(parent1, shift, axes);
    roll3->set_friendly_name("roll3");

    const auto result = std::make_shared<ngraph::opset1::Result>(roll3);
    result->set_friendly_name("result");

    return std::make_shared<ov::Model>(
        ngraph::ResultVector{ result },
        parameter2 == nullptr ? ParameterVector{ parameter1 } : ParameterVector{ parameter1, parameter2 },
        "SubgraphTransposeMatMulFunction");
}

std::shared_ptr<Model> SubgraphTransposeMatMulFunction::initOriginal() const {
    return get(input_shapes, precision, transpose, mat_mul);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

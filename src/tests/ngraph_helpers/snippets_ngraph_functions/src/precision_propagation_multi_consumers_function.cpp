// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_propagation_multi_consumers_function.hpp"

#include "subgraph_converts.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/convert_truncation.hpp>
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

//std::shared_ptr<ngraph::Function> PrecisionPropagationMultiConsumerFunction::get(
//    const ngraph::element::Type precision1,
//    const ngraph::PartialShape& inputShape1,
//    const ngraph::element::Type precision2,
//    const ngraph::PartialShape& inputShape2,
//    const ngraph::element::Type constant_precision,
//    const std::pair<element::Type, element::Type>& convertion_before_op1,
//    const element::Type convertion_before_op2_1,
//    const std::pair<element::Type, element::Type>& convertion_before_op2_2,
//    const element::Type convertion_after_op2) {
//    const auto parameter1 = std::make_shared<ngraph::opset1::Parameter>(precision1, inputShape1);
//    parameter1->set_friendly_name("parameter1");
//
//    std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::Minimum>(
//        parameter1,
//        std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}));
//    parent->set_friendly_name("minimum");
//
//    const auto parameter2 = std::make_shared<ngraph::opset1::Parameter>(precision2, inputShape2);
//    parameter2->set_friendly_name("parameter2");
//
//    std::shared_ptr<Node> node1 = std::make_shared<ngraph::opset1::Maximum>(
//        parent,
//        std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}));
//    node1->set_friendly_name("maximum1");
//
//    std::shared_ptr<Node> node2 = std::make_shared<ngraph::opset1::Add>(parent, parameter2);
//    node2->set_friendly_name("add");
//
//    std::shared_ptr<Node> node3 = std::make_shared<ngraph::opset1::Maximum>(
//        parent,
//        std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}));
//    node3->set_friendly_name("maximum2");
//
//    const auto result1 = std::make_shared<ngraph::opset1::Result>(node1);
//    result1->set_friendly_name("result1");
//
//    const auto result2 = std::make_shared<ngraph::opset1::Result>(node2);
//    result2->set_friendly_name("result2");
//
//    const auto result3 = std::make_shared<ngraph::opset1::Result>(node3);
//    result3->set_friendly_name("result3");
//
//    const ngraph::ResultVector results{ result1, result2, result3 };
//    const ngraph::ParameterVector parameters{ parameter1, parameter2 };
//    const auto model = std::make_shared<ngraph::Function>(results, parameters, "PrecisionPropagationMultiConsumerFunction");
//    model->validate_nodes_and_infer_types();
//    return model;
//}

}  // namespace snippets
}  // namespace test
}  // namespace ov

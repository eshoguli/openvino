// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "function_helper.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
std::vector<std::shared_ptr<Node>> FunctionHelper::makePrerequisitesOriginal() {
    std::vector<std::shared_ptr<Node>> nodes;

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>();
    parameter->set_friendly_name("parameter");
    nodes.push_back(parameter);

    const auto convert1 = std::make_shared<ngraph::opset1::Convert>(parameter, ov::element::u8);
    convert1->set_friendly_name("convert1");
    nodes.push_back(convert1);

    const auto relu1 = std::make_shared<ngraph::opset1::Relu>(convert1);
    relu1->set_friendly_name("relu1");
    nodes.push_back(relu1);

    const auto convert2 = std::make_shared<ngraph::opset1::Convert>(relu1, ov::element::f32);
    convert2->set_friendly_name("convert2");
    nodes.push_back(convert2);

    const auto slope2 = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{-1.f});
    const auto relu2 = std::make_shared<ngraph::opset1::PRelu>(convert2, slope2);
    relu2->set_friendly_name("relu2");
    nodes.push_back(relu2);

    return nodes;
}

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
std::vector<std::shared_ptr<Node>> FunctionHelper::makePrerequisitesReference() {
    std::vector<std::shared_ptr<Node>> nodes;

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>();
    parameter->set_friendly_name("parameter");
    nodes.push_back(parameter);

    const auto convert1 = std::make_shared<ngraph::opset1::Convert>(parameter, ov::element::u8);
    convert1->set_friendly_name("convert1");
    nodes.push_back(convert1);

    const auto relu1 = std::make_shared<ngraph::opset1::Relu>(convert1);
    relu1->set_friendly_name("relu1");
    nodes.push_back(relu1);

    const auto convert2 = std::make_shared<ngraph::opset1::Convert>(relu1, ov::element::f32);
    convert2->set_friendly_name("convert2");
    nodes.push_back(convert2);

    auto getSubgraphBody = []() {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>();
        parameter->set_friendly_name("parameter");

        const auto slope = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{-1.f});
        const auto relu = std::make_shared<ngraph::opset1::PRelu>(parameter, slope);
        relu->set_friendly_name("relu");

        const auto result = std::make_shared<ngraph::opset1::Result>(relu);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ngraph::ParameterVector{ parameter }, "SubgraphWithFakeQuantizeBody");
    };

    const auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(OutputVector{ convert2 }, getSubgraphBody());

    subgraph->set_friendly_name("subgraph");
    nodes.push_back(subgraph);

    return nodes;
}

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
std::vector<std::shared_ptr<Node>> FunctionHelper::makePrerequisites() {
    std::vector<std::shared_ptr<Node>> nodes;

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>();
    parameter->set_friendly_name("parameter");
    nodes.push_back(parameter);

    const auto convert1 = std::make_shared<ngraph::opset1::Convert>(parameter, ov::element::u8);
    convert1->set_friendly_name("convert1");
    nodes.push_back(convert1);

    const auto relu1 = std::make_shared<ngraph::opset1::Relu>(convert1);
    relu1->set_friendly_name("relu1");
    nodes.push_back(relu1);

    const auto convert2 = std::make_shared<ngraph::opset1::Convert>(relu1, ov::element::f32);
    convert2->set_friendly_name("convert2");
    nodes.push_back(convert2);

    return nodes;
}

std::shared_ptr<Node> FunctionHelper::applyPrerequisites(const std::shared_ptr<Node>& parent, const std::vector<std::shared_ptr<Node>>& prerequisites) {
    std::shared_ptr<ngraph::Node> currentParent;
    if (prerequisites.empty()) {
        currentParent = parent;
    }
    else {
        auto begin = prerequisites[0];
        if (is_type<ngraph::opset1::Parameter>(begin)) {
            begin = prerequisites[1];
        }
        begin->set_argument(0, parent);

        currentParent = *prerequisites.rbegin();
    }
    return currentParent;
}

std::shared_ptr<Node> FunctionHelper::getSubgraph(const std::shared_ptr<Model>& f, const int index) {
    int currentIndex = 0;
    std::shared_ptr<ngraph::snippets::op::Subgraph> subgraph;
    for (const auto& op : f->get_ordered_ops()) {
        auto tmp_subgraph = as_type_ptr<ngraph::snippets::op::Subgraph>(op);
        if (tmp_subgraph != nullptr) {
            if (index == currentIndex) {
                return tmp_subgraph;
            }
            subgraph = tmp_subgraph;
            currentIndex++;
        }
    }

    if (index != -1) {
        return nullptr;
    }
    return subgraph;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph/opsets/opset1.hpp"
#include "snippets/op/convert_saturation.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationMultiConsumerFunction {
public:
    template<typename T>
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision1,
        const ngraph::PartialShape& inputShape1,
        const ngraph::element::Type precision2,
        const ngraph::PartialShape& inputShape2,
        const ngraph::element::Type constant_precision,
        const element::Type convertion_after_parameter1_all = {},
        const element::Type convertion_after_parameter1_min_max = {},
        const element::Type convertion_after_parameter2 = {},
        const element::Type convertion_before_maximum = {},
        const element::Type convertion_after_maximum = {},
        const element::Type convertion_before_minimum = {},
        const element::Type convertion_after_minimum = {},
        const element::Type convertion_before_add = {},
        const element::Type convertion_after_add = {}) {
        const auto parameter1 = std::make_shared<ngraph::opset1::Parameter>(precision1, inputShape1);
        parameter1->set_friendly_name("parameter1");
        std::shared_ptr<Node> parent1 = create_convert(parameter1, convertion_after_parameter1_all);

        //std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::Minimum>(
        //    parameter1,
        //    std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}));
        //parent->set_friendly_name("minimum");
        //std::shared_ptr<Node> parent = std::dynamic_pointer_cast<Node>(parameter1);

        const auto parameter2 = std::make_shared<ngraph::opset1::Parameter>(precision2, inputShape2);
        parameter2->set_friendly_name("parameter2");
        std::shared_ptr<Node> parent2 = create_convert(parameter2, convertion_after_parameter2);

        std::shared_ptr<Node> parent_min_max = create_convert(parent1, convertion_after_parameter1_min_max);
        std::shared_ptr<Node> maximum_parent = create_convert(parent_min_max, convertion_before_maximum);
        std::shared_ptr<Node> maximum = std::make_shared<ngraph::opset1::Maximum>(
            maximum_parent,
            create_convert(
                std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}),
                precision1 == maximum_parent->output(0).get_element_type() ? element::undefined : maximum_parent->output(0).get_element_type()));
        maximum->set_friendly_name("maximum");
        maximum = create_convert(maximum, convertion_after_maximum);

        std::shared_ptr<Node> minimum_parent = create_convert(parent_min_max, convertion_before_minimum);
        std::shared_ptr<Node> minimum = std::make_shared<ngraph::opset1::Maximum>(
            minimum_parent,
            create_convert(
                std::make_shared<ngraph::opset1::Constant>(precision1, ov::Shape{}, std::vector<float>{1.f}),
                precision1 == minimum_parent->output(0).get_element_type() ? element::undefined : minimum_parent->output(0).get_element_type()));
        minimum->set_friendly_name("minimum");
        minimum = create_convert(minimum, convertion_after_minimum);

        std::shared_ptr<Node> add_parent1 = create_convert(parent1, convertion_before_add);
        std::shared_ptr<Node> add = std::make_shared<T>(add_parent1, parent2);
        add->set_friendly_name("add");
        add = create_convert(add, convertion_after_add);

        const auto result1_1 = std::make_shared<ngraph::opset1::Result>(maximum);
        result1_1->set_friendly_name("result1_1");
        const auto result1_2 = std::make_shared<ngraph::opset1::Result>(maximum);
        result1_2->set_friendly_name("result1_2");

        const auto result2 = std::make_shared<ngraph::opset1::Result>(add);
        result2->set_friendly_name("result2");

        const auto result3 = std::make_shared<ngraph::opset1::Result>(minimum);
        result3->set_friendly_name("result3");

        const ngraph::ResultVector results{ result1_1, result1_2, result2, result3 };
        const ngraph::ParameterVector parameters{ parameter1, parameter2 };
        const auto model = std::make_shared<ngraph::Function>(results, parameters, "PrecisionPropagationMultiConsumerFunction");
        model->validate_nodes_and_infer_types();
        return model;
    }

private:
    static std::shared_ptr<Node> create_convert(std::shared_ptr<Node> parent, const element::Type convertion_type) {
        return convertion_type == element::undefined
                   ? std::dynamic_pointer_cast<Node>(parent)
                   : std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convertion_type);
    }

    static std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ov::Node>> make_branch(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const size_t index,
        const element::Type convertion_type) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        parameter->set_friendly_name("parameter" + std::to_string(index));

        std::shared_ptr<Node> parent = create_convert(parameter, convertion_type);

        return {parameter, parent};
    }
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

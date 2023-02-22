// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/precision_propagation_multi_consumers.hpp"

#include <gtest/gtest.h>
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "common_test_utils/common_utils.hpp"
#include "precision_propagation_multi_consumers_function.hpp"
#include "ngraph/pass/serialize.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
class DummyAdd : public ngraph::opset1::Add {
public:
    OPENVINO_OP("DummyAdd", "test::snippets");

    DummyAdd(
        const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : ngraph::opset1::Add(arg0, arg1, auto_broadcast) {
        constructor_validate_and_infer_types();
    }

    DummyAdd(const ngraph::opset1::Add& add)
        : Add(add.get_input_source_output(0), add.get_input_source_output(1), add.get_autob()) {
        constructor_validate_and_infer_types();
    }

    DummyAdd() = default;

    void validate_and_infer_types() override {
        const auto input_type1 = get_input_element_type(0);
        const auto input_type2 = get_input_element_type(1);

        const element::Type output_type = (input_type1 == element::i8) || (input_type2 == element::i8) ?
            element::i32 :
            get_input_element_type(0);

        set_output_type(0, output_type, get_input_partial_shape(0));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<DummyAdd>(new_args.at(0), new_args.at(1), this->get_autob());
    }
};

class DummyPrecisionPropogationTargetMachine : public DummyTargetMachine {
public:
    DummyPrecisionPropogationTargetMachine(
        const std::set<std::vector<element::Type>>& supported_precisions_maximum,
        const std::set<std::vector<element::Type>>& supported_precisions_minimum,
        const std::set<std::vector<element::Type>>& supported_precisions_add)
        : DummyTargetMachine() {
        jitters[DummyAdd::get_type_info_static()] = ngraph::snippets::jitters_value {
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            [&supported_precisions_add](const std::shared_ptr<ngraph::Node>& n) { return supported_precisions_add; }};

        jitters[op::v1::Maximum::get_type_info_static()] = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            [&supported_precisions_maximum](const std::shared_ptr<ngraph::Node>&n) { return supported_precisions_maximum; }};

        jitters[op::v1::Minimum::get_type_info_static()] = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            [&supported_precisions_minimum](const std::shared_ptr<ngraph::Node>& n) { return supported_precisions_minimum; } };

        auto default_jitter = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            [](const std::shared_ptr<ngraph::Node>& n) { return std::set<std::vector<element::Type>>{};} };
        jitters[ngraph::snippets::op::ConvertSaturation::get_type_info_static()] = default_jitter;
    }
};

} // namespace

std::string PrecisionPropagationMultiConsumerTest::getTestCaseName(testing::TestParamInfo<PrecisionPropagationMultiConsumerParams> obj) {
    std::pair<Shape, Shape> shapes;
    PrecisionPropagationMultiConsumerParamsValues test_values;
    std::tie(shapes, test_values) = obj.param;

    auto to_string = [](const std::set<std::vector<element::Type>>& precisions_pack) noexcept {
        auto to_string = [](const std::vector<element::Type>& precisions) noexcept {
            std::ostringstream result;
            result << "{";
            for (const auto& precision : precisions) {
                result << precision << "_";
            }
            result << "}";
            return result.str();
        };

        std::ostringstream result;
        result << "{";
        for (const auto& precisions : precisions_pack) {
            result << to_string(precisions) << "_";
        }
        result << "}";
        return result.str();
    };

    std::ostringstream result;
    result << "IN0_" << shapes.first << "_" << test_values.input_types[0] << "_"
           << "IN1_" << shapes.second << "_" << test_values.input_types[1] << "_"
           << "IN2_" << test_values.input_types[2]
           << to_string(test_values.actual.supported_precisions_maximum) << "_"
           << to_string(test_values.actual.supported_precisions_minimum) << "_"
           << to_string(test_values.actual.supported_precisions_add) << "_"
           << test_values.expected.convertion_after_parameter1_all << "_"
           << test_values.expected.convertion_after_parameter1_min_max << "_"
           << test_values.expected.convertion_after_parameter2 << "_"
           << test_values.expected.convertion_before_maximum << "_"
           << test_values.expected.convertion_after_maximum << "_"
           << test_values.expected.convertion_before_minimum << "_";
    return result.str();
}

TEST_P(PrecisionPropagationMultiConsumerTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    function = PrecisionPropagationMultiConsumerFunction::get<DummyAdd>(
        test_values.input_types[0],
        shapes.first,
        test_values.input_types[1],
        shapes.second,
        test_values.input_types[2],
        test_values.actual.convertion_after_parameter1_all,
        test_values.actual.convertion_after_parameter1_min_max,
        test_values.actual.convertion_after_parameter2,
        test_values.actual.convertion_before_maximum,
        test_values.actual.convertion_after_maximum,
        test_values.actual.convertion_before_minimum,
        test_values.actual.convertion_after_minimum,
        test_values.actual.convertion_before_add,
        test_values.actual.convertion_after_add);

    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
    ngraph::pass::Serialize("svg/test.actual.xml", "svg/test.actual.bin").run_on_model(function);

    const auto target_machine =
        std::make_shared<DummyPrecisionPropogationTargetMachine>(
            test_values.actual.supported_precisions_maximum,
            test_values.actual.supported_precisions_minimum,
            test_values.actual.supported_precisions_add);
    ngraph::snippets::pass::PropagatePrecision(element::f32, target_machine).run_on_model(function);
    function->validate_nodes_and_infer_types();

    ngraph::pass::VisualizeTree("svg/test.transformed.svg").run_on_model(function);
    ngraph::pass::Serialize("svg/test.transformed.xml", "svg/test.transformed.bin").run_on_model(function);

    function_ref = PrecisionPropagationMultiConsumerFunction::get<DummyAdd>(
        test_values.input_types[0],
        shapes.first,
        test_values.input_types[1],
        shapes.second,
        test_values.input_types[2],
        test_values.expected.convertion_after_parameter1_all,
        test_values.expected.convertion_after_parameter1_min_max,
        test_values.expected.convertion_after_parameter2,
        test_values.expected.convertion_before_maximum,
        test_values.expected.convertion_after_maximum,
        test_values.expected.convertion_before_minimum,
        test_values.expected.convertion_after_minimum,
        test_values.expected.convertion_before_add,
        test_values.expected.convertion_after_add);

    ngraph::pass::VisualizeTree("svg/test.reference.svg").run_on_model(function_ref);
    ngraph::pass::Serialize("svg/test.reference.xml", "svg/test.reference.bin").run_on_model(function_ref);

    comparator.disable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

namespace PrecisionPropagationMultiConsumersTestInstantiation {
// clang-format off

std::vector<std::pair<Shape, Shape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<PrecisionPropagationMultiConsumerParamsValues> test_cases {
    {
        {element::i8, element::i8, element::i8},
        {
            {{element::f32, element::f32}},
            {{element::f32, element::f32}},
            {{element::f32, element::f32}, {element::i32, element::i32}},
            {},
            {},
            {}
        },
        {
            {},             // convertion_after_parameter1_all
            {element::f32}, // convertion_after_parameter1_min_max
            {element::i32}, // convertion_after_parameter2
            {},             // convertion_before_maximum
            {element::i8},  // convertion_after_maximum
            {},             // convertion_before_minimum
            {element::i8},  // convertion_after_minimum
            {element::i32}, // convertion_before_add
            {}              // convertion_after_add
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_PrecisionPropagationMultiConsumerTest,
    PrecisionPropagationMultiConsumerTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(test_cases)),
    PrecisionPropagationMultiConsumerTest::getTestCaseName);

// clang-format on
} // namespace PrecisionPropagationMultiConsumersTestInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov

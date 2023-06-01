// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pull_eltwise_through_data_movement.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pass/serialize.hpp"


ov::pass::PullEltwiseThroughDataMovement::PullEltwiseThroughDataMovement() {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMov);
    auto eltwise_pattern = ngraph::pattern::wrap_type<ov::op::v1::Multiply, ov::op::v0::FakeQuantize>();

    ov::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();

        if ((eltwise->get_friendly_name() != "/roberta/encoder/layer.0/attention/output/dense/MatMul/fq_input_0") &&
            (eltwise->get_friendly_name() != "/roberta/encoder/layer.0/attention/self/Reshape_3/sq_mul")) {
            return false;
        }

        std::cout << "PullEltwiseThroughDataMovement: " << eltwise->get_type_name() << ": " << eltwise->get_friendly_name() << std::endl;

        if (transformation_callback(eltwise)) {
            return false;
        }

        bool was_updated = false;

        while (true) {
            auto parent_output = eltwise->get_input_source_output(0);
            if (!ov::is_type<ov::op::v1::Reshape>(parent_output.get_node()) && !ov::is_type<ov::op::v1::Transpose>(parent_output.get_node())) {
                return was_updated;
            }

            auto parent = parent_output.get_node_shared_ptr();
            auto parent_input = parent->input(0);
            auto parent2_output = parent->get_input_source_output(0);

            auto child_input = *eltwise->get_output_target_inputs(0).begin();

            if (ov::is_type<ngraph::op::util::BinaryElementwiseArithmetic>(eltwise)) {
                if (ov::is_type<ov::op::v1::Reshape>(parent)) {
                    //OutputVector output_values(eltwise->get_output_size());
                    //OutputVector input_values = {};

                    //auto backprop_reshape = std::make_shared<ov::op::v1::Reshape>(
                    //    std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{ 4 }, {0, 0, 12,64}));
                    //auto new_const = parent->constant_fold(output_values, input_values);

                    const auto values = ov::as_type_ptr<ov::op::v0::Constant>(eltwise->get_input_node_shared_ptr(1))->cast_vector<float>();
                    const auto new_const = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{ 1, 1, 12, 64 }, values);
                    //const auto new_const = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{ 1 }, std::vector<float>{1.0});
                    replace_node(eltwise->get_input_node_shared_ptr(1), new_const);
                }
                if (ov::is_type<ov::op::v1::Transpose>(parent)) {
                    //OutputVector output_values(eltwise->get_output_size());
                    //OutputVector input_values = {};

                    //auto backprop_reshape = std::make_shared<ov::op::v1::Reshape>(
                    //    std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{ 4 }, {0, 0, 12,64}));
                    //auto new_const = parent->constant_fold(output_values, input_values);

                    const auto values = ov::as_type_ptr<ov::op::v0::Constant>(eltwise->get_input_node_shared_ptr(1))->cast_vector<float>();
                    const auto new_const = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{ 1, 12, 1, 64 }, values);
                    //const auto new_const = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{ 1 }, std::vector<float>{1.0});
                    replace_node(eltwise->get_input_node_shared_ptr(1), new_const);
                }
            }

            parent_output.remove_target_input(eltwise->input(0));
            child_input.replace_source_output(parent_output);

            parent2_output.remove_target_input(parent_input);
            eltwise->input(0).replace_source_output(parent2_output);

            eltwise->output(0).remove_target_input(child_input);
            parent_input.replace_source_output(eltwise->output(0));

            eltwise->validate_and_infer_types();
            was_updated = true;
        }
        return was_updated;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}

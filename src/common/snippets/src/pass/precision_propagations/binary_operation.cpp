// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/precision_propagations/binary_operation.hpp"

#include <snippets/itt.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/utils.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "ngraph/op/util/op_types.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include "ov_ops/type_relaxed.hpp"

#include <ngraph/rt_info.hpp>

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

ngraph::snippets::pass::precision_propagation::BinaryOperation::BinaryOperation(const std::vector<std::pair<ov::element::Type, ov::element::Type>>& precisions) : precisions(precisions) {
    MATCHER_SCOPE(AddTransformation);
    auto wrapper1 = ngraph::pattern::wrap_type<ngraph::snippets::op::ConvertSaturation>(pattern::consumers_count(1));
    auto wrapper2 = ngraph::pattern::wrap_type<ngraph::snippets::op::ConvertSaturation>(pattern::consumers_count(1));
    auto matcher = ngraph::pattern::wrap_type<opset1::Add>({ wrapper1, wrapper2 });

    ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
        auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        // TODO: need tests
        if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(root)) {
            return false;
        }

        auto op = std::dynamic_pointer_cast<opset1::Add>(root);
        if (!op) {
            throw ngraph_error("unexpected operation type");
        }

//#ifdef CPU_DEBUG_CAPS_SNIPPETS
//        ngraph::pass::VisualizeTree("svg/snippets.precision_propagation.add.1.svg").run_on_model(m);
//#endif

        //for (const auto& input : op->inputs()) {
        //    auto convert = input.get_source_output().get_node_shared_ptr();
        //    auto parameter = convert->input(0).get_source_output().get_node_shared_ptr();

        //    const auto dependencies = convert->get_control_dependencies();
        //}


        ov::element::TypeVector input_precisions;
        for (const auto& input : op->inputs()) {
            input_precisions.push_back(input.get_source_output().get_element_type());
        }

        //std::vector<element::Type> output_precisions = { input_precisions[0]};
        ov::element::TypeVector output_precisions = { op->output(0).get_element_type() };

        auto op_relaxed = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(*op, input_precisions, output_precisions);
        op_relaxed->set_overridden_output_type(op->get_input_source_output(0).get_node()->input(0).get_source_output().get_element_type());
        // TODO: do we need it right now?
        //std::dynamic_pointer_cast<ngraph::Node>(op_relaxed)->validate_and_infer_types();
        copy_runtime_info(op, op_relaxed);

        for (auto input_index = 0ull; input_index < op->get_input_size(); input_index++) {
            const auto& input = op->input(input_index);

            auto convert = input.get_source_output().get_node_shared_ptr();
            convert->output(0).remove_target_input(input);

            auto parent = convert->input(0).get_source_output().get_node_shared_ptr();
            parent->output(0).remove_target_input(convert->input(0));

            op_relaxed->input(input_index).replace_source_output(parent->output(0));
        }

        auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(op_relaxed, output_precisions[0]);
        replace_node(op, convert);

//#ifdef CPU_DEBUG_CAPS_SNIPPETS
//        ngraph::pass::VisualizeTree("svg/snippets.precision_propagation.add.2.svg").run_on_model(m);
//#endif

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

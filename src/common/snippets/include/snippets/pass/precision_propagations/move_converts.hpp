// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>
#include <snippets/itt.hpp>
#include "snippets/op/convert_saturation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagations {

using precisions_set = std::set<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>>;

// TODO: not used
template<typename Operation>
class MoveConvert: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BinaryOperation", "0");
    MoveConvert(const precisions_set precisions = {}, const size_t input_count_ = 0ul) {
        MATCHER_SCOPE(Operation);

        size_t input_count;
        if (input_count_ == 0ull) {
            input_count = precisions.size() == 0ull ? 1ull : precisions.begin()->first.size();
        } else {
            input_count = input_count_;
        }

        assert(std::all_of(
            precisions.begin(),
            precisions.end(),
            [&precisions](const std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>& pair) { return precisions.begin()->first.size() == pair.first.size(); }));

        OutputVector parents(
            input_count,
            ngraph::pattern::wrap_type<ngraph::snippets::op::ConvertSaturation>(pattern::consumers_count(1)));

        auto matcher = ngraph::pattern::wrap_type<Operation>(parents);

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto root = m.get_match_root();
            if (transformation_callback(root)) {
                return false;
            }

            // TODO: need tests
            if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(root)) {
                return false;
            }

            auto op = std::dynamic_pointer_cast<Operation>(root);
            if (!op) {
                throw ngraph_error("unexpected operation type");
            }

            ov::element::TypeVector input_precisions;
            for (const auto& input : op->inputs()) {
                input_precisions.push_back(input.get_source_output().get_element_type());
            }

            ov::element::TypeVector output_precisions = { op->output(0).get_element_type() };

            auto op_relaxed = std::make_shared<ngraph::op::TypeRelaxed<Operation>>(*op, input_precisions, output_precisions);
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

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
        this->register_matcher(m, callback);
    }
private:
    const std::vector<std::pair<ov::element::Type, ov::element::Type>> precisions;
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

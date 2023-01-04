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

template<typename Operation>
class PropagatePrecision: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagatePrecision", "0");
    PropagatePrecision(const precisions_set precisions = {}, const size_t input_count_ = 0ul) {
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

        auto matcher = ngraph::pattern::wrap_type<Operation>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto root = m.get_match_root();
            if (transformation_callback(root)) {
                return false;
            }

            // TODO: need tests
            if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(root)) {
                // precision was propagated
                return true;
            }

            auto op = std::dynamic_pointer_cast<Operation>(root);
            if (!op) {
                throw ngraph_error("unexpected operation type");
            }

            ov::element::TypeVector input_precisions;
            for (const auto& input : op->inputs()) {
                input_precisions.push_back(input.get_source_output().get_element_type());
            }

            //auto output_precision = element::undefined;
            //for (const auto& precision_item : precisions) {
            //    precision_item.
            //}

            // TODO: not completed
            ov::element::TypeVector output_precisions = { op->output(0).get_element_type() };

            auto op_relaxed = std::make_shared<ngraph::op::TypeRelaxed<Operation>>(*op, input_precisions, output_precisions);
            op_relaxed->set_overridden_output_type(op->get_input_source_output(0).get_node()->input(0).get_source_output().get_element_type());
            // TODO: do we need it right now?
            //std::dynamic_pointer_cast<ngraph::Node>(op_relaxed)->validate_and_infer_types();
            copy_runtime_info(op, op_relaxed);
            replace_node(op, op_relaxed);

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

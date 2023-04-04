// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_transformations/move_transpose_through_convert.hpp"

#include <assert.h>
#include <memory>

#include "ov_ops/type_relaxed.hpp"
#include "snippets/itt.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/transpose_decomposition.hpp"

using namespace ngraph;
using namespace ov::intel_cpu::pass;

MoveTransposeThroughConvert::MoveTransposeThroughConvert(const element::Type source, const element::Type target) :
    source(source),
    target(target) {
    MATCHER_SCOPE(MoveTransposeThroughConvert);
    auto transpose_wrapper = ngraph::pattern::wrap_type<ngraph::opset1::Transpose>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto root = m.get_match_root();
        const auto transpose = ov::as_type_ptr<ngraph::opset1::Transpose>(root);
        assert(transpose);

        const auto const_orders = ov::as_type_ptr<ngraph::opset1::Constant>(transpose->get_input_node_shared_ptr(1));
        assert(const_orders);
        const auto orders = const_orders->get_vector<int>();
        if (snippets::pass::TransposeDecomposition::supported_cases.find(orders) == snippets::pass::TransposeDecomposition::supported_cases.end()) {
            return false;
        }

        const auto convert = transpose->get_input_node_shared_ptr(0);
        if (!ngraph::is_type<ngraph::snippets::op::ConvertSaturation>(convert) || (convert->get_output_element_type(0) != source)) {
            return false;
        }

        const auto& parameter = convert->get_input_node_shared_ptr(0);
        if (!ngraph::is_type<ngraph::op::Parameter>(parameter) || (parameter->get_output_element_type(0) != target)) {
            return false;
        }

        // remove convert
        convert->output(0).replace(parameter->output(0));

        const auto new_convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(transpose, convert->get_output_element_type(0));
        ngraph::copy_runtime_info(convert, new_convert);

        const auto& output = transpose->output(0);
        for (auto& input : output.get_target_inputs()) {
            auto child = input.get_node();
            if (child == new_convert.get()) {
                continue;
            }

            input.replace_source_output(new_convert->output(0));


            if (ngraph::is_type<ngraph::op::Result>(input.get_node())) {
                input.get_tensor_ptr()->add_names(output.get_tensor_ptr()->get_names());

                const std::string original_name = transpose->get_friendly_name();
                transpose->set_friendly_name(original_name + "_original");
                new_convert->set_friendly_name(original_name);
            }
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_wrapper, matcher_name);
    register_matcher(m, callback);
}

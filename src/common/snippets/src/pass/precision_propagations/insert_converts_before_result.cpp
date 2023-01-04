// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/precision_propagations/insert_converts_before_result.hpp"

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

ngraph::snippets::pass::precision_propagations::InsertConvertsBeforeResults::InsertConvertsBeforeResults(const ov::element::Type supported_precision) {
    MATCHER_SCOPE(AddTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::Parameter>();

    ngraph::graph_rewrite_callback callback = [this, supported_precision](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        const auto output_type = op->output(0).get_element_type();
        if (supported_precision != output_type) {
            for (auto& output : op->outputs()) {
                for (auto& input : output.get_target_inputs()) {
                    auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(output, supported_precision);
                    output.remove_target_input(input);
                    input.replace_source_output(convert->output(0));
                }
            }
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

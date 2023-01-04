// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/precision_propagations/keep_precision.hpp"

#include <assert.h>
#include <memory>

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

ngraph::snippets::pass::precision_propagations::KeepPrecision::KeepPrecision(const ov::element::Type precision) {
    MATCHER_SCOPE(KeepPrecision);

    default_pass_callback callback = [precision](const std::shared_ptr<Node>& node) {
        assert(!std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(node));

        if (ngraph::is_type<ngraph::opset1::Parameter>(node) || ngraph::is_type<ngraph::opset1::Result>(node)) {
            return false;
        }

        for (auto input_index = 0ull; input_index < node->get_input_size(); input_index++) {
            const auto& input = node->input(input_index);
            if (input.get_source_output().get_element_type() == precision) {
                continue;
            }

            auto parent_output = input.get_source_output();
            parent_output.remove_target_input(input);

            auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent_output, precision);

            input.replace_source_output(convert->output(0));
        }

        return true;
    };

    this->register_callback(callback);
}

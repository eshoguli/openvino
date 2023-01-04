// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/sequential_manager.hpp"
//#include "snippets/pass/precision_propagations/sequential_graph_rewrite.hpp"

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

ngraph::snippets::pass::precision_propagations::SequentialManager::SequentialManager() {
}

void ngraph::snippets::pass::precision_propagations::SequentialManager::run_passes(std::shared_ptr<ov::Model> func) {
    for (const auto& node : func->get_ordered_ops()) {
        auto at_least_once_was_handled = false;
        for (auto& pass : m_pass_list) {
            auto matcher_pass = std::dynamic_pointer_cast<ov::pass::MatcherPass>(pass);
            SequentialGraphRewrite graph_rewrite(matcher_pass);
            const auto result = graph_rewrite.apply_matcher_pass(func, node);
            if (result) {
                at_least_once_was_handled = result;

                // TODO: refactor later: validate and infer types for neccessary nodes only
                func->validate_nodes_and_infer_types();
                break;
            }

            //auto function_changed = ov::pass::GraphRewrite(matcher_pass).run_on_model(func);
            // apply_matcher_passes
        }

        if (!at_least_once_was_handled) {
            for (const auto& pass : m_default_pass_list) {
                const auto result = pass->get_callback()(node);
                if (result) {
                    // TODO: refactor later: validate and infer types for neccessary nodes only
                    func->validate_nodes_and_infer_types();
                }
            }
        }
    }
}
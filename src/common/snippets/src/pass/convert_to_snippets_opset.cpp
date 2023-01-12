// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convert_to_snippets_opset.hpp"

#include <ngraph/rt_info.hpp>
#include "snippets/itt.hpp"
#include "snippets/op/add.hpp"

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

#define REPLACE_AND_CONTINUE(instance, type_from, type_to)                   \
    if (is_type<type_from>(instance)) {\
    auto new_instance = std::make_shared<type_to>(*as_type_ptr<type_from>(instance));\
    replace_node(instance, new_instance);\
    copy_runtime_info(instance, new_instance);\
    was_updated = true;\
    continue;\
}\

bool ngraph::snippets::pass::ConvertToSnippetsOpset::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertToSnippetsOpset);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertToSnippetsOpset")

    bool was_updated = false;
    const auto& ops = f->get_ordered_ops();
    for (const auto& op : f->get_ordered_ops()) {
        REPLACE_AND_CONTINUE(op, ngraph::opset1::Add, ngraph::snippets::op::Add)
    }

    return was_updated;
}

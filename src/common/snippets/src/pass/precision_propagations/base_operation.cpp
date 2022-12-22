// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/precision_propagations/base_operation.hpp"

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

ngraph::snippets::pass::precision_propagation::BaseOperation::BaseOperation(const ov::element::Type exec_type) : exec_type(exec_type) {
    MATCHER_SCOPE(AddTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::Add>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

//#ifdef CPU_DEBUG_CAPS_SNIPPETS
//        ngraph::pass::VisualizeTree("svg/snippets.precision_propagation.add.1.svg").run_on_model(m);
//#endif

        // TODO:

//#ifdef CPU_DEBUG_CAPS_SNIPPETS
//        ngraph::pass::VisualizeTree("svg/snippets.precision_propagation.add.2.svg").run_on_model(m);
//#endif

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

void ngraph::snippets::pass::precision_propagation::BaseOperation::insert_convert(const std::shared_ptr<Node>& parent) {
    //
}
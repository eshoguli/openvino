// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/add.hpp"

#include <assert.h>
#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/op/add.hpp"

ngraph::snippets::pass::Add::Add() {
    MATCHER_SCOPE(Add);
    auto matcher = ngraph::pattern::wrap_type<opset1::Add>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        assert(is_type<ngraph::opset1::Add>(op));

        const auto autob = as_type_ptr<ngraph::opset1::Add>(op)->get_autob();
        auto new_add = std::make_shared<snippets::op::Add>(op->get_input_source_output(0), op->get_input_source_output(1), autob);
        replace_node(op, new_add);
        copy_runtime_info(op, new_add);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

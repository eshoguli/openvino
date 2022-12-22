// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/precision_propagations/precision_propagation.hpp"

#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>
#include "snippets/snippets_isa.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/utils.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "ngraph/op/util/op_types.hpp"

#include <ngraph/rt_info.hpp>

#include "snippets/pass/precision_propagations/add.hpp"

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

ngraph::snippets::pass::precision_propagation::PrecisionPropagation::PrecisionPropagation(const ov::element::Type exec_type) : exec_type(exec_type) { }

bool ngraph::snippets::pass::precision_propagation::PrecisionPropagation::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_FUNCTION_SCOPE(PrecisionPropagation);

#ifdef CPU_DEBUG_CAPS_SNIPPETS
    ngraph::pass::VisualizeTree("svg/snippets.low_precision.1.svg").run_on_model(m);
#endif

    auto passConfig = get_pass_config();
    ngraph::pass::Manager manager(passConfig);
    manager.register_pass<ngraph::snippets::pass::precision_propagation::Add>();
    manager.run_passes(m);

#ifdef CPU_DEBUG_CAPS_SNIPPETS
    ngraph::pass::VisualizeTree("svg/snippets.low_precision.2.svg").run_on_model(m);
#endif

    return true;
}

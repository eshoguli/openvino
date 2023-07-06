// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/if.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

// TODO: debug only
#include "ngraph/pass/serialize.hpp"
#include "ngraph/pass/visualize_tree.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

IfTransformation::IfTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(MaxPoolTransformation);
    //auto matcher = pattern::wrap_type<ov::opset8::If>({ pattern::wrap_type<opset1::Multiply>() });
    auto matcher = pattern::wrap_type<ov::opset8::If>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool IfTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    //const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    //if (dequantization.empty()) {
    //    return false;
    //}

    return true;
}

bool IfTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    // TODO: debug only
    ngraph::pass::Serialize("svg/lpt.if_1.xml", "svg/lpt.if_1.bin").run_on_model(ov::Model::m_global_model);
    ngraph::pass::VisualizeTree("svg/lpt.if_1.svg").run_on_model(ov::Model::m_global_model);

    //if (!canBeTransformed(context, m.get_match_root())) {
    //    return false;
    //}

    const auto root = ov::as_type_ptr<ov::opset8::If>(m.get_match_root());
    assert(root != nullptr);

    for (const auto& input : root->inputs()) {
        const auto& dequantization = NetworkHelper::getDequantization(root, defaultPrecisions, input.get_index());
        if (dequantization.empty()) {
            continue;
        }

        const auto& input_node = input.get_source_output().get_node_shared_ptr();
        const auto& descriptions = root->get_input_descriptions(input.get_index());
    }

    ngraph::pass::Serialize("svg/lpt.if_then_1.xml", "svg/lpt.if_then_1.bin").run_on_model(root->get_then_body());
    ngraph::pass::VisualizeTree("svg/lpt.if_then_1.svg").run_on_model(root->get_then_body());

    ngraph::pass::Serialize("svg/lpt.if_else_1.xml", "svg/lpt.if_else_1.bin").run_on_model(root->get_else_body());
    ngraph::pass::VisualizeTree("svg/lpt.if_else_1.svg").run_on_model(root->get_else_body());

    //const std::shared_ptr<Node> node = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    //moveDequantizationAfter(context, node, NetworkHelper::getDequantization(node, defaultPrecisions), false);

    // TODO: debug only
    ngraph::pass::Serialize("svg/lpt.if_then_2.xml", "svg/lpt.if_then_2.bin").run_on_model(root->get_then_body());
    ngraph::pass::VisualizeTree("svg/lpt.if_then_2.svg").run_on_model(root->get_then_body());

    ngraph::pass::Serialize("svg/lpt.if_else_2.xml", "svg/lpt.if_else_2.bin").run_on_model(root->get_else_body());
    ngraph::pass::VisualizeTree("svg/lpt.if_else_2.svg").run_on_model(root->get_else_body());

    ngraph::pass::Serialize("svg/lpt.if_2.xml", "svg/lpt.if_2.bin").run_on_model(ov::Model::m_global_model);
    ngraph::pass::VisualizeTree("svg/lpt.if_2.svg").run_on_model(ov::Model::m_global_model);

    return true;
}

bool IfTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph

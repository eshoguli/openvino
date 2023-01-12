// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/propagate_precision.hpp"

#include <assert.h>
#include <memory>
#include <ie_ngraph_utils.hpp>
#include <snippets/itt.hpp>

//#include "snippets/snippets_isa.hpp"
//#include "snippets/op/convert_saturation.hpp"
//#include "snippets/utils.hpp"
//#include "ov_ops/type_relaxed.hpp"
//#include "ngraph/op/util/op_types.hpp"
//#include "ie_precision.hpp"

//#include <ngraph/pattern/op/wrap_type.hpp>
//#include "ov_ops/type_relaxed.hpp"

#include <ngraph/rt_info.hpp>

#ifdef CPU_DEBUG_CAPS_SNIPPETS
#include "ngraph/pass/visualize_tree.hpp"
#endif

namespace {
void validate_and_infer_types_below(const std::shared_ptr<ov::Node>& node) {
    std::unordered_set<ov::Node*> handled;

    std::deque<ov::Node*> nodes;
    nodes.push_back(node.get());

    while (!nodes.empty()) {
        ov::Node* n = nodes.back();
        nodes.pop_back();

        n->revalidate_and_infer_types();

        for (const auto& output : n->outputs()) {
            for (const auto& source_input : output.get_target_inputs()) {
                const auto& source_node = source_input.get_node();
                nodes.push_back(source_node);
            }
        }
    }
}
} // namespace

ngraph::snippets::pass::PropagatePrecision::PropagatePrecision(
    const ov::element::Type supported_precision,
    const std::shared_ptr<const TargetMachine>& target_machine) : supported_precision(supported_precision), target_machine(target_machine) {
}

bool ngraph::snippets::pass::PropagatePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagatePrecision")

    const auto& ops = f->get_ordered_ops();
    for (const auto& op : f->get_ordered_ops()) {
        auto type_info = op->get_type_info();
        if (!target_machine->has(type_info)) {
            // TODO: throw exception here
        }

        std::vector<InferenceEngine::Precision> input_precisions;
        for (const auto& input : op->inputs()) {
            const auto input_precision = input.get_source_output().get_element_type();
            // TODO: how to convert: InferenceEngine::Precision => ov::element::Type ?
            const auto input_precision_ie = InferenceEngine::details::convertPrecision(input_precision);
            input_precisions.push_back(input_precision_ie);
        }
        
        const auto supported_precisions = target_machine->get_supported_precisions(type_info);

        assert(
            std::all_of(
                supported_precisions.begin(), 
                supported_precisions.end(),
                [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) { return precisions.size() == input_precisions.size(); }));

        if (!supported_precisions.empty() &&
            !std::any_of(
                supported_precisions.begin(),
                supported_precisions.end(),
                [&input_precisions](const std::vector<InferenceEngine::Precision>& precisions) { return precisions == input_precisions; })) {
            for (const auto& input : op->inputs()) {
                auto parent_output = input.get_source_output();
                parent_output.remove_target_input(input);

                auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent_output, supported_precision);
                input.replace_source_output(convert->output(0));
                
                validate_and_infer_types_below(op);
            }
        }
    }

    return false;
}

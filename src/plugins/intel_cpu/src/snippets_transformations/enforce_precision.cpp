// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_transformations/enforce_precision.hpp"

#include <assert.h>
#include <memory>

#include "ov_ops/type_relaxed.hpp"
#include "snippets/itt.hpp"
#include "ngraph/rt_info.hpp"
#include "snippets/pass/propagate_precision.hpp"

using namespace ngraph;
using namespace ov::intel_cpu::pass;

EnforcePrecision::EnforcePrecision(
    const element::Type source,
    const element::Type target,
    const bool is_target_isa_supported,
    std::function<EnforcePrecision::Operation(const std::shared_ptr<ngraph::Node>& op)> get_supported_precisions) :
    source(source),
    target(target),
    is_target_isa_supported(is_target_isa_supported),
    get_supported_precisions(get_supported_precisions == nullptr ? get_supported_precisions_default : get_supported_precisions) {
}

bool EnforcePrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(PropagatePrecision);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EnforcePrecision")

    bool was_updated = false;
    for (const auto& op : f->get_ordered_ops()) {
        const auto& op_desc = get_supported_precisions(op);

        if (op_desc.precisions.empty()) {
            continue;
        }

        if (op_desc.require_target_isa && !is_target_isa_supported) {
            continue;
        }

        std::vector<element::Type> actual_precisions;
        for (const auto& input : op->inputs()) {
            actual_precisions.push_back(input.get_element_type());
        }

        auto op_is_appropriate = true;

        std::vector<element::Type> supported_precisions_to_enforce;
        for (const auto& supported_precisions : op_desc.precisions) {
            if (supported_precisions.size() != actual_precisions.size()) {
                continue;
            }

            auto op_has_to_be_handled = false;
            for (auto index = 0ull; index < supported_precisions.size(); ++index) {
                // check operation position
                if ((supported_precisions[index] == target) && op_desc.only_after_parameter) {
                    const auto& convert = op->get_input_node_shared_ptr(index);
                    if (!ngraph::is_type<ngraph::snippets::op::ConvertSaturation>(convert) || (convert->get_output_element_type(0) != source)) {
                        op_is_appropriate = false;
                        break;
                    }

                    const auto& parameter = convert->get_input_node_shared_ptr(0);
                    if (!ngraph::is_type<ngraph::op::Parameter>(parameter) || (parameter->get_output_element_type(0) != target)) {
                        op_is_appropriate = false;
                        break;
                    }
                }

                // check input precisions
                if ((supported_precisions[index] == target) && (actual_precisions[index] == source)) {
                    op_has_to_be_handled = true;
                } else {
                    // current input is not required to be enforced
                    if ((supported_precisions[index] != element::undefined) && (supported_precisions[index] != actual_precisions[index])) {
                        op_is_appropriate = false;
                        break;
                    }
                    continue;
                }
            }
            if (!op_is_appropriate) {
                break;
            }

            if (op_has_to_be_handled) {
                supported_precisions_to_enforce = supported_precisions;
                break;
            }
        }

        if (!op_is_appropriate || supported_precisions_to_enforce.empty()) {
            continue;
        }

        const auto insert_convert = [](
            const Output<Node>& parent_output,
            const std::shared_ptr<Node>& op,
            const size_t input_index,
            const element::Type& target) {
                auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(
                    parent_output,
                    target);
                ngraph::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);
                op->set_argument(input_index, convert);
        };

        for (auto index = 0ull; index < supported_precisions_to_enforce.size(); ++index) {
            if ((supported_precisions_to_enforce[index] == target) || (actual_precisions[index] == source)) {
                const auto op_parent = ov::as_type_ptr<ngraph::snippets::op::ConvertSaturation>(op->get_input_node_shared_ptr(index));
                if ((op_parent != nullptr) && (op_parent->get_input_element_type(0) == target)) {
                    // remove convert
                    op_parent->output(0).replace(op_parent->get_input_source_output(0));
                    was_updated = true;
                } else if (supported_precisions_to_enforce[index] != actual_precisions[index]) {
                    insert_convert(op->get_input_source_output(index), op, index, target);
                    was_updated = true;
                }
            }
        }

        auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(op);
        if (was_updated || (type_relaxed_node != nullptr)) {
            const bool res = ngraph::snippets::pass::PropagatePrecision::validate_and_infer_types_and_restore_outputs(op);
            was_updated = was_updated || res;
        }
    }

    return was_updated;
}

EnforcePrecision::Operation EnforcePrecision::get_supported_precisions_default(
    const std::shared_ptr<ngraph::Node>&op) noexcept {
    if (ov::is_type<ngraph::opset1::Transpose>(op)) {
        return EnforcePrecision::Operation{ true, false, {{element::bf16, element::undefined}} };
    }

    if (ov::is_type<ngraph::snippets::op::Brgemm>(op)) {
        return EnforcePrecision::Operation{ false, true, {{element::bf16, element::bf16}} };
    }


    return {};
}

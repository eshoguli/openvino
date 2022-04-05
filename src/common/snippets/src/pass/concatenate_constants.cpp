// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/concatenate_constants.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/utils/utils.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "ngraph/pass/visualize_tree.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::ConcatenateConstants, "Snippets::ConcatenateConstants", 0);

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
void parameters_to_constants(std::shared_ptr<ov::Model>& body, const std::unordered_map<std::string, std::shared_ptr<opset1::Constant>>& constant_input_ids) {
    auto ops = body->get_ops();
    for (auto& op : ops) {
        auto parameter = as_type_ptr<ngraph::opset1::Parameter>(op);
        if (parameter == nullptr) {
            continue;
        }

        auto it = constant_input_ids.find(parameter->get_friendly_name());
        if (it == constant_input_ids.end()) {
            continue;
        }

        const auto& subgraph_constant = it->second;
        auto body_constant = subgraph_constant->clone_with_new_inputs({});

        body_constant->set_friendly_name(parameter->get_friendly_name());
        for (auto input : parameter->output(0).get_target_inputs()) {
            input.replace_source_output(body_constant->output(0));
        }

        body->remove_parameter(parameter);
    }
    body->validate_nodes_and_infer_types();
}

void constants_to_parameters(
    std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph,
    std::shared_ptr<ov::Model>& body,
    const std::unordered_map<std::string, std::shared_ptr<opset1::Parameter>>& parameter_input_ids) {
    std::vector<ngraph::Output<Node>> new_inputs;
    new_inputs.reserve(subgraph->get_input_size());
    for (auto i = 0; i < subgraph->get_input_size(); ++i) {
        auto input = subgraph->get_input_source_output(i);
        if (!is_type<opset1::Constant>(input.get_node_shared_ptr())) {
            new_inputs.push_back(input);
        }
    }

    auto ops = body->get_ops();
    for (auto& op : ops) {
        auto constant = as_type_ptr<ngraph::opset1::Constant>(op);
        if ((constant == nullptr) || (ngraph::shape_size(constant->get_output_shape(0)) == 1ul)) {
            continue;
        }

        new_inputs.push_back(constant);
        auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());

        parameter->set_friendly_name(constant->get_friendly_name());
        for (auto input : constant->output(0).get_target_inputs()) {
            input.replace_source_output(parameter->output(0));
        }

        body->add_parameters(ParameterVector{parameter});
    }
    body->validate_nodes_and_infer_types();

    const auto new_subgraph = subgraph->clone_with_new_inputs(new_inputs);
    replace_node(subgraph, new_subgraph);

    new_subgraph->set_friendly_name(subgraph->get_friendly_name());
    copy_runtime_info(subgraph, new_subgraph);
}

} // namespace

ConcatenateConstants::ConcatenateConstants() {
    auto wrapper = ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ConcatenateConstants");

        auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->get_body();
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming2").run_on_model(body);

        // TODO: first version limitation
        const size_t axis = 1;
        std::vector<size_t> split_lengths;

        std::unordered_map<std::string, std::shared_ptr<opset1::Constant>> constant_input_ids;
        OutputVector previousInputs;
        OutputVector outputs;
        std::unordered_map<std::string, std::shared_ptr<opset1::Parameter>> parameter_input_ids;
        std::unordered_map<std::string, size_t> indexByFriendlyName;
        std::string concatenatedConstantFriendlyName = "";
        bool concatenatedConstantWasAdded = false;
        ngraph::element::Type inputPrecision;

        for (auto i = 0;  i < subgraph->get_input_size(); ++i) {
            auto target_input = subgraph->input(i);
            auto source_output = target_input.get_source_output();
            auto constant = as_type_ptr<opset1::Constant>(source_output.get_node()->shared_from_this());

            // TODO: first version: 1) add shape limitation check 2) per 1 axis only 2) no precision check

            if (constant != nullptr) {
                if (!concatenatedConstantWasAdded) {
                    concatenatedConstantFriendlyName = constant->get_friendly_name();
                    concatenatedConstantWasAdded = true;
                    inputPrecision = constant->output(0).get_element_type();
                }
                const auto& shape = constant->output(0).get_shape();
                const size_t split_length = shape.size() >= (axis + 1ul) ? shape[axis] : 1ul;
                split_lengths.push_back(shape[axis]);

                constant->output(0).remove_target_input(target_input);

                constant_input_ids.emplace(constant->get_friendly_name(), constant);
                outputs.push_back(constant->output(0));
                indexByFriendlyName[constant->get_friendly_name()] = split_lengths.size() - 1;
                continue;
            } else {
                previousInputs.push_back(source_output);
            }
        }

        OutputVector foldingOutputs(1);
        auto concat = std::make_shared<ngraph::opset1::Concat>(outputs, axis);
        if (!concat->constant_fold(foldingOutputs, outputs)) {
            throw ngraph_error("constants were not concatenated");
        }
        const auto result = ov::as_type_ptr<opset1::Constant>(foldingOutputs[0].get_node_shared_ptr());
        if (result == nullptr) {
            throw ngraph_error("constants were not concatenated");
        }

        result->set_friendly_name(concatenatedConstantFriendlyName);
        previousInputs.push_back(result);



        auto parameter = std::make_shared<opset1::Parameter>(inputPrecision, result->output(0).get_partial_shape());
        parameter->set_friendly_name(concatenatedConstantFriendlyName);

        std::shared_ptr<Node> split =
            std::all_of(split_lengths.begin(), split_lengths.end(), [&split_lengths](const size_t value) { return split_lengths[0] == value; }) ?
                std::dynamic_pointer_cast<Node>(std::make_shared<opset1::Split>(
                    parameter,
                    std::make_shared<opset1::Constant>(element::i32, Shape{}, std::vector<size_t>{axis}),
                    split_lengths.size())) :
                std::make_shared<opset1::VariadicSplit>(
                    parameter,
                    std::make_shared<opset1::Constant>(element::i32, Shape{ 1ul }, std::vector<size_t>{axis}),
                    std::make_shared<opset1::Constant>(element::i32, Shape{ split_lengths.size() }, split_lengths));

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_model(body);
        body->validate_nodes_and_infer_types();

        auto parameters = body->get_parameters();
        for (auto i = 0; i < parameters.size(); ++i) {
            auto parameter = parameters[i];
            auto it = constant_input_ids.find(parameter->get_friendly_name());
            if (it == constant_input_ids.end()) {
                continue;
            }

            auto indexIt = indexByFriendlyName.find(parameter->get_friendly_name());
            if (indexIt == indexByFriendlyName.end()) {
                throw ngraph_error("parameter was not mapped");
            }

            //auto tragetInputs1 = parameter->output(0).get_target_inputs();

            for (auto input : parameter->output(0).get_target_inputs()) {
                input.replace_source_output(split->output(indexIt->second));
            }

            body->remove_parameter(parameter);
            std::cout << "removed: " << parameter->get_friendly_name() << std::endl;

            //auto tragetInputs2 = parameter->output(0).get_target_inputs();

            //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_model(body);
            //body->validate_nodes_and_infer_types();
        }
        body->add_parameters({ parameter });

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_model(body);
        body->validate_nodes_and_infer_types();

        auto newSubgraph = std::make_shared<ngraph::snippets::op::Subgraph>(previousInputs, body);
        //auto newSubgraph = subgraph->clone_with_new_inputs(previousInputs);
        replace_node(subgraph, newSubgraph);
        copy_runtime_info(subgraph, newSubgraph);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_model(body);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(wrapper, "snippets::pass::ConcatenateConstants");
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
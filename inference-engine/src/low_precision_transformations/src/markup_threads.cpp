// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_threads.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/rt_info/thread_attribute.hpp"
#include "low_precision/network_helper.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupThreads, "MarkupThreads", 0);

// markup consumers
bool ngraph::pass::low_precision::MarkupThreads::run_on_function(std::shared_ptr<ngraph::Function> f) {
    //auto get_current_thread = [](const std::shared_ptr<Node>& node) -> size_t {
    //    auto& rt = node->get_rt_info();
    //    auto it = rt.find(ngraph::VariantWrapper<ThreadAttribute>::type_info.name);
    //    if (it == rt.end()) {
    //        return nullptr;
    //    }
    //
    //    auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<ThreadAttribute>>(it->second);
    //    assert(attribute != nullptr);
    //    return attribute;
    //};

    auto are_several_consumers = [](const std::shared_ptr<Node>& node){
        const size_t outputs_size = node->get_output_size();
        size_t output_index = 0;
        size_t input_index = 0;
        for (; output_index < outputs_size;) {
            auto output = node->output(output_index);
            auto consumer_inputs = output.get_target_inputs();
            for (auto& consumer_input : consumer_inputs) {
                auto consumer_node = consumer_input.get_node()->shared_from_this();
                if (ngraph::is_type<opset1::Result>(consumer_node) || ngraph::pass::low_precision::isBranchConcatenation(consumer_node)) {
                    continue;
                }

                if ((output_index != 0) || (input_index != 0)) {
                    return true;
                }
                input_index++;
            }
            output_index++;
        }
        return false;
    };

    size_t thread_id = 0ul;
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (transformation_callback(node) || as_type_ptr<opset1::Constant>(node)) {
            continue;
        }

        if (as_type_ptr<opset1::Parameter>(node)) {
            thread_id++;
            auto& rt = node->get_rt_info();
            rt[ngraph::VariantWrapper<ThreadAttribute>::type_info.name] = std::make_shared<ngraph::VariantWrapper<ThreadAttribute>>(ThreadAttribute(thread_id));
        }

        //if (node->get_friendly_name() == "bottleneck2_0/dim_red/conv/fq_input_0") {
        //    std::cout << "" << std::endl;
        //}
        //if (node->get_friendly_name() == "bottleneck2_0/add") {
        //    std::cout << "" << std::endl;
        //}
        //std::cout << node->get_type_name() << ": " << node->get_friendly_name() << std::endl;

        auto attribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(node);
        if (attribute == nullptr) {
            continue;
        }
        const bool several_consumers = are_several_consumers(node);

        //if (node->get_friendly_name() == "bottleneck2_1/add/fq_input_1") {
        //    std::cout << "" << std::endl;
        //}

        for (auto output : node->outputs()) {
            auto consumer_inputs = output.get_target_inputs();
            for (auto& consumer_input : consumer_inputs) {
                auto consumer_node = consumer_input.get_node()->shared_from_this();
                if (is_type<opset1::Result>(consumer_node)) {
                    continue;
                }

                //if (consumer_node->get_friendly_name() == "bottleneck1_1/add/fq_input_0") {
                //    std::cout << "TO DEBUG" << std::endl;
                //}

                const size_t current_thread_id = several_consumers ? ++thread_id : attribute->get().thread_id;

                auto consumer_node_attribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(consumer_node);
                if (consumer_node_attribute != nullptr) {
                    //// node has been handled before
                    //if (consumer_node_attribute->get().input_thread_ids.size() == 1ul) {
                    //    // second thread is entering the node: update existing thread id
                    //    ++thread_id;
                    //    consumer_node_attribute->get().thread_id = thread_id;
                    //    attribute->get().output_thread_ids.insert(thread_id);
                    //}
                    consumer_node_attribute->get().input_thread_ids.insert(current_thread_id);
                    continue;
                }

                auto new_consumer_node_attribute = std::make_shared<ngraph::VariantWrapper<ThreadAttribute>>(ThreadAttribute(
                    current_thread_id,
                    attribute->get().thread_id));
                auto& rt = consumer_node->get_rt_info();
                rt[ngraph::VariantWrapper<ThreadAttribute>::type_info.name] = new_consumer_node_attribute;
                attribute->get().output_thread_ids.insert(current_thread_id);

                if (ngraph::pass::low_precision::isBranchConcatenation(consumer_node)
                    //&& (new_consumer_node_attribute->get().input_thread_ids.size() > 1ul)
                    ) {
                    new_consumer_node_attribute->get().completion_counter = std::make_shared<CompletionCounter>(consumer_node->get_input_size());
                }
            }
        }
    }
    return true;
}

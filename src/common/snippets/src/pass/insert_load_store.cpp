// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/insert_load_store.hpp"

#include <assert.h>

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertLoad::InsertLoad(const size_t count) {
    MATCHER_SCOPE(InsertLoad);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Parameter>(), matcher_name),
            [this, count](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertLoad")
            auto root = m.get_match_root();

            const auto& inputs = root->get_output_target_inputs(0);
            if (inputs.size() == 1ul) {
                const auto input = inputs.begin();
                const auto& input_node = input->get_node();

                // TODO: workaround: not completed
                // if (is_type<opset1::MaxPool>(input_node)) {
                if (is_type<opset1::MaxPool>(input_node) ||
                    // Convolution data & Convolution weights
                    is_type<ngraph::opset1::Convolution>(input_node) ||
                    // Convolution biases
                    (is_type<ngraph::opset1::Add>(input_node) &&
                     is_type<ngraph::opset1::Convolution>(input_node->get_input_node_shared_ptr(0))) ||

                    // GroupConvolution weights (GroupConvolution data has to be excluded)
                    is_type<ngraph::opset1::GroupConvolution>(input_node) ||
                    // GroupConvolution biases
                    (is_type<ngraph::opset1::Add>(input_node) &&
                     is_type<ngraph::opset1::GroupConvolution>(input_node->get_input_node_shared_ptr(0)))) {
#ifdef CPU_DEBUG_CAPS_SNIPPETS
                    std::cout << "InsertLoad: was skipped" << std::endl;
#endif
                    return false;
                }

//                // TODO: workaround: not completed
//                if (is_type<ov::intel_cpu::ConvolutionMerged1x1Kernel>(input_node) ||
//                    is_type<ngraph::snippets::op::ConvolutionMergedDwKernel>(input_node)) {
//#ifdef CPU_DEBUG_CAPS_SNIPPETS
//                    std::cout << "InsertLoad: was skipped" << std::endl;
//#endif
//                    return false;
//                }
            }

            // check if already has Load as an output
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (ov::is_type<ngraph::snippets::op::Load>(consumer.get_node())) {
                        return false;
                    }
                }
            }

            // TODO: workaround for ConvolutionKernel weights support
            assert(root->get_output_size() == 1ul);
            assert(root->output(0).get_target_inputs().size() == 1ul);
            const auto child_input = *root->output(0).get_target_inputs().begin();
            const bool empty =
                (child_input.get_index() == 1ul) && (is_type<ngraph::opset1::Convolution>(child_input.get_node()));

            auto load = std::make_shared<ngraph::snippets::op::Load>(root, empty);
            ngraph::copy_runtime_info(root, load);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != load) {
                        consumer.replace_source_output(load);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        });
}

ngraph::snippets::pass::InsertStore::InsertStore(const size_t count) {
    MATCHER_SCOPE(InsertStore);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Result>(), matcher_name),
            [this, count](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertStore")
            auto root = m.get_match_root();

            // check if already has Store as an input
            for (auto input : root->inputs()) {
                if (ov::is_type<ngraph::snippets::op::Store>(input.get_source_output().get_node())) {
                    return false;
                }
            }

            auto store = std::make_shared<ngraph::snippets::op::Store> (root->input_value(0), count);
            ngraph::copy_runtime_info(root, store);
            root->set_argument(0, store);
            return true;
        });
}

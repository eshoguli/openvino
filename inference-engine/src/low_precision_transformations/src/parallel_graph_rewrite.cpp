// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/parallel_graph_rewrite.hpp"
#include <ie_parallel.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/thread_attribute.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ParallelGraphRewrite, "ngraph::pass::ParallelGraphRewrite", 0);

bool ngraph::pass::low_precision::ParallelGraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Initialize execution queue with nodes in topological order
    deque<std::shared_ptr<Node>> nodes_to_run;
    for (auto& node : f->get_ordered_ops()) {
        nodes_to_run.emplace_back(node);
    }
    return apply_matcher_passes(f, std::move(nodes_to_run));
}

bool ngraph::pass::low_precision::ParallelGraphRewrite::apply_matcher_passes(shared_ptr<Function> f, deque<std::shared_ptr<Node>> nodes_to_run) {
    //OV_ITT_SCOPED_TASK(itt::domains::nGraph, "pass::GraphRewrite::run_on_function");

    bool rewritten = false;
    const auto& pass_config = get_pass_config();

    // Check that all Matchers in MatcherPasses has type bases root node
    bool all_roots_has_type = true;
    std::unordered_map<NodeTypeInfo, std::vector<size_t>> type_to_matcher;
    for (size_t matcher_index = 0; matcher_index < m_matchers.size(); ++matcher_index) {
        // Skip passes that are disabled
        if (pass_config->is_disabled(m_matchers[matcher_index]->get_type_info()))
            continue;

        auto matcher = m_matchers[matcher_index]->get_matcher();
        if (!matcher) {
            all_roots_has_type = false;
            break;
        }

        auto root = matcher->get_pattern_value().get_node_shared_ptr();
        // pattern::op::AnyOutput operation automatically appends for multi output operations inside
        // Matcher and to gen actual root node we need to take it's parent.
        if (auto any_type = dynamic_pointer_cast<pattern::op::AnyOutput>(root)) {
            root = any_type->input_value(0).get_node_shared_ptr();
        }

        // if root is an operation from opset or has pattern::op::WrapType type then we can extract
        // it's type
        // and use it in unordered_map as key for fast MatcherPass search. Otherwise type is unknown
        // and default algorithm is used.
        if (auto p = dynamic_pointer_cast<pattern::op::Pattern>(root)) {
            if (auto any_type = dynamic_pointer_cast<pattern::op::WrapType>(p)) {
                for (const auto& root_type_info : any_type->get_wrapped_types()) {
                    type_to_matcher[root_type_info].push_back(matcher_index);
                }
            } else {
                all_roots_has_type = false;
                break;
            }
        } else {
            type_to_matcher[root->get_type_info()].push_back(matcher_index);
        }

        // TODO: traverse parents for root_type_info in order to register complete list of matchers
        // including ones triggered by parent type info.
    }

    // This lambda preforms execution of particular MatcherPass on given node.
    // It automatically handles nodes registered by MatcherPass during transformation and set
    // transformation callback.
    auto run_matcher_pass = [&](std::shared_ptr<MatcherPass> m_pass, std::shared_ptr<Node> node) -> bool {
        // Keep this property check for backward compatibility. In future transformation property
        // will be deprecated and removed.
        if (m_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && f->is_dynamic()) {
            //NGRAPH_DEBUG << "matcher callback requires static shape but the "
            //                "function is dynamic, skipping this "
            //                "optimization till the shapes are fully "
            //                "materialized";
            return false;
        }

        // Apply MatcherPass. In case if it returns true no other MatcherPasses will apply
        // to this node
        bool status = m_pass->apply(node);

        // In case if MatcherPass registered nodes they will be added to the beginning of execution
        // queue
        const auto& new_nodes = m_pass->get_new_nodes();
        if (!new_nodes.empty()) {
            // Need to push nodes in reverse order as we expect that nodes in new_nodes
            // vector are in topological order
            for (auto it = new_nodes.rbegin(); it != new_nodes.rend(); it++) {
                nodes_to_run.emplace_front(*it);
            }
            m_pass->clear_new_nodes();
        }
        return status;
    };

    // list of matchers to run for a node; define here to keep memory allocated
    std::vector<size_t> matcher_passes_to_run;

    while (!nodes_to_run.empty()) {
        auto node = nodes_to_run.front();
        nodes_to_run.pop_front();

        //auto attribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(node);
        //if (attribute != nullptr) {
        //    std::cout << node->get_type_name() << "." << node->get_friendly_name() << ": thread_id: " << attribute->get().thread_id << std::endl;
        //}

        // Recursive apply Matchers for sub-graph based nodes
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_function(sub_graph);
            }
        }
        // Temporary keep this GraphRewrite property for backward compatibility
        if (m_enable_shape_inference) {
            node->revalidate_and_infer_types();
        }
        // If all Matchers in MatcherPasses has type based root node then we apply efficient
        // algorithm for finding matchers
        if (all_roots_has_type) {
            const DiscreteTypeInfo* node_type_info = &node->get_type_info();
            matcher_passes_to_run.clear();
            while (node_type_info) {
                auto matchers = type_to_matcher.find(*node_type_info);
                if (matchers != type_to_matcher.end()) {
                    // do not run found matchers immediately, need to collect all matchers for
                    // parents
                    // and sort them in order of the registration
                    matcher_passes_to_run.insert(matcher_passes_to_run.end(),
                                                 matchers->second.begin(),
                                                 matchers->second.end());
                }
                node_type_info = node_type_info->parent;
            }

            std::sort(matcher_passes_to_run.begin(), matcher_passes_to_run.end());

            // TODO: type_to_matcher with just collected list of matchers to enable
            // fast processing at the next time when node with the same type will be processed

            for (size_t matcher_index : matcher_passes_to_run) {
                if (run_matcher_pass(m_matchers[matcher_index], node)) {
                    rewritten = true;
                    break;
                }
            }
        } else {
            // Otherwise we use default algorithm that iterates over all registered matcher passes
            for (auto& m_pass : m_matchers) {
                // Skip passes that are disabled
                if (pass_config->is_disabled(m_pass->get_type_info()))
                    continue;

                if (run_matcher_pass(m_pass, node)) {
                    rewritten = true;
                    break;
                }
            }
        }
    }
    return rewritten;
}
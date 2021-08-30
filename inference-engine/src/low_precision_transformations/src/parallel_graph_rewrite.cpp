// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/parallel_graph_rewrite.hpp"
#include <ie_parallel.hpp>
#include <thread>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/thread_attribute.hpp"
#include "unistd.h"

#include "tbb/task_group.h"

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

//struct ApplyMatcherPassesParams {
//    ngraph::pass::low_precision::ParallelGraphRewrite* graphRewrite;
//    shared_ptr<Function> f;
//    deque<std::shared_ptr<Node>> nodes_to_run;
//};

std::shared_ptr<Node> ngraph::pass::low_precision::ParallelGraphRewrite::fill_ordered_ops_for_thread_execution(
    const std::shared_ptr<Node>& node,
    std::deque<std::shared_ptr<Node>>& nodes_to_run) {
    auto childNode = node->shared_from_this();
    while (true) {
        if (ngraph::pass::low_precision::isBranchConcatenation(childNode)) {
            return childNode;
        }
        nodes_to_run.push_back(childNode);
        auto outputs = childNode->outputs();
        auto children = outputs[0].get_target_inputs();
        if (children.empty()) {
            return childNode;
        }
        childNode = children.begin()->get_node()->shared_from_this();
    }
    return nullptr;
}

// TODO: up to down traversal with assumption:
//   1) stop if branch is splitted: splitted branch will be handled separated threads
//   2) based on thread markup attribute output threads
std::shared_ptr<Node> ngraph::pass::low_precision::ParallelGraphRewrite::fill_ordered_ops_for_main_execution(
    const std::shared_ptr<Node>& node,
    std::deque<std::shared_ptr<Node>>& nodes_to_run) {
    auto getChild = [](std::set<Input<Node>> children) -> std::shared_ptr<Node> {
        std::shared_ptr<Node> selectedChild;
        for (const auto child : children) {
            auto childNode = child.get_node()->shared_from_this();
            if (!ngraph::pass::low_precision::isBranchConcatenation(childNode)) {
                assert(selectedChild == nullptr);
                selectedChild = childNode;
            }
        }
        return selectedChild;
    };

    auto childNode = node->shared_from_this();
    while (true) {
        // FIXME: LPT: there is issue here: operations can be skipped
        nodes_to_run.push_back(childNode);

        if ((childNode->get_output_size() >= 1ul) &&
            ((childNode->get_output_size() > 1ul) || (childNode->output(0).get_target_inputs().size() > 1ul))) {
            auto attribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(childNode);
            if ((attribute != nullptr) && (attribute->get().output_thread_ids.size() > 1ul)) {
                // TODO: add first child
                auto firstChild = childNode->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                nodes_to_run.push_back(firstChild);
                return childNode;
            }
        }

        auto outputs = childNode->outputs();
        auto children = outputs[0].get_target_inputs();
        if (children.empty()) {
            return childNode;
        }
        if (children.size() == 1ul) {
            childNode = children.begin()->get_node()->shared_from_this();
        } else {
            childNode = getChild(children);
        }
        if (childNode == nullptr) {
            return nullptr;
        }
    }
    return nullptr;
}

void print(deque<std::shared_ptr<Node>> nodes_to_run, const std::string& title) {
    std::stringstream  ss;
    ss << title << ", nodes_to_run (" << std::this_thread::get_id() << "): " << nodes_to_run.size() << ":" << std::endl;
    size_t number = 1ul;
    for (auto it : nodes_to_run) {
        auto& nodeToPrint = *it;
        ss << "\t" << number << ": " << nodeToPrint.get_friendly_name() << " (" << nodeToPrint.get_type_name() << ")" << std::endl;
        number++;
    }
    std::cout << ss.str();
}

bool ngraph::pass::low_precision::ParallelGraphRewrite::apply_matcher_passes_in_thread(
    shared_ptr<Function> f,
    std::shared_ptr<Node>& node) {
    bool rewritten = false;
    const auto& threadAttribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(node);
    if ((threadAttribute != nullptr) && (!threadAttribute->get().handled)) {
        threadAttribute->get().handled = true;
        tbb::task_group g;

#ifdef DEBUG_THREADING
        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/poc/cpu.before.svg").run_on_function(f);
        std::stringstream  ss;
        ss << "main thread before parallelization (" <<
            std::this_thread::get_id() << "): " <<
            node->get_friendly_name() << " (" << node->get_type_name() << ")" <<
            std::endl;
        std::cout << ss.str();
#endif

        const auto inputs = node->output(0).get_target_inputs();
        for (auto& input : inputs) {
            deque<std::shared_ptr<Node>> nodes_to_run_for_thread_execution;
            auto syncNode = fill_ordered_ops_for_thread_execution(input.get_node()->shared_from_this(), nodes_to_run_for_thread_execution);
            //if (is_type<op::v0::Result>(syncNode)) {
            //    continue;
            //}

#ifdef DEBUG_THREADING
            std::stringstream ss2;
            ss2 << "apply_matcher_passes_in_thread (" <<
                std::this_thread::get_id() << "): sync node=" <<
                syncNode->get_friendly_name() << " (" << syncNode->get_type_name() << "): " <<
                nodes_to_run_for_thread_execution.size() <<
                std::endl;
            std::cout << ss2.str();

            //if (syncNode->get_friendly_name() == "bottleneck3_6/add") {
            //    std::cout << "JUST TO DEBUG" << std::endl;
            //}
#endif

#ifndef THREAD_BY_BRANCH
            if (nodes_to_run_for_thread_execution.empty()) {
                continue;
            }
#endif
            g.run([this, f, nodes_to_run_for_thread_execution, syncNode]() {
#ifndef THREAD_BY_BRANCH
                assert(!nodes_to_run_for_thread_execution.empty());
#endif
                if (nodes_to_run_for_thread_execution.empty()) {
#ifdef DEBUG_THREADING
                    std::stringstream ss;
                    ss << "thread was started (" <<
                        std::this_thread::get_id() << ", " <<
                        "nodes: " << nodes_to_run_for_thread_execution.size() << ")" << std::endl;
                    std::cout << ss.str();
#endif
                } else {
#ifdef DEBUG_THREADING
                    auto deque_node = nodes_to_run_for_thread_execution.front();
                    std::stringstream ss;
                    ss << "thread was started (" <<
                        std::this_thread::get_id() << ", " <<
                        "nodes: " << nodes_to_run_for_thread_execution.size() << "): " <<
                        deque_node->get_friendly_name() << " (" << deque_node->get_type_name() << ")" << std::endl;
                    std::cout << ss.str();
#endif

                    // TODO: debug only
                    //print(nodes_to_run_for_thread_execution);
                    this->apply_matcher_passes(f, nodes_to_run_for_thread_execution);
                }
#ifdef DEBUG_THREADING
                std::stringstream ss2;
                    ss2 << "thread was completed (" <<
                    std::this_thread::get_id() << "): " <<
                    syncNode->get_friendly_name() << " (" << syncNode->get_type_name() << ")" <<
                    std::endl;
                std::cout << ss2.str();
#endif

                if (syncNode == nullptr) {
#ifdef DEBUG_THREADING
                    // TODO: the last node - not handled yet
                    std::stringstream ss3;
                    ss3 << "last node was achieved (" << std::this_thread::get_id() << ")" << std::endl;
                    std::cout << ss3.str();
#endif
                } else {
                    auto attributeWrapper = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(syncNode);
                    assert(is_type<op::v0::Result>(syncNode) || (!is_type<op::v0::Result>(syncNode) && (attributeWrapper != nullptr)));
                    if (attributeWrapper != nullptr) {
                        auto& attribute = attributeWrapper->get();
                        assert(is_type<op::v0::Result>(syncNode) || (!is_type<op::v0::Result>(syncNode) && (attribute.completion_counter != nullptr)));
                        if ((attribute.completion_counter != nullptr) && attribute.completion_counter->complete()) {
#ifdef DEBUG_THREADING
                            std::cout << "threads completed (" << std::this_thread::get_id() << ")" << std::endl << std::endl;
                            ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/poc/cpu.after.svg").run_on_function(f);
#endif

                            // each thread can achieve completed node
                            deque<std::shared_ptr<Node>> nodes_to_run_for_main_execution;
                            fill_ordered_ops_for_main_execution(syncNode, nodes_to_run_for_main_execution);

#ifdef DEBUG_THREADING
                            print(nodes_to_run_for_main_execution, "main execution");
#endif

                            this->apply_matcher_passes(f, nodes_to_run_for_main_execution);
                        }
                    }
                }
            });
        }
        g.wait();
    }

    return rewritten;
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

        GraphRewriteContext context;

        // Apply MatcherPass. In case if it returns true no other MatcherPasses will apply
        // to this node
        bool status = m_pass->apply(node, &context);

#ifdef DEBUG_THREADING
        if (status) {
            // operation double handling check
            std::stringstream keyStream;
            keyStream << m_pass->get_type_info().name << "/" << node->get_friendly_name() << "/" << node->get_type_name();
            const auto key = keyStream.str();
            assert(handled.find(key) == handled.end());
            handled.insert(key);
        }
#endif

        //if (status) {
        //    std::stringstream ss;
        //    ss << "\tapply_matcher_passes (" << std::this_thread::get_id() << "): " <<
        //        m_pass->get_type_info().name << ": " <<
        //        node->get_friendly_name() << " (" << node->get_type_name() << ")" <<
        //        std::endl;
        //    std::cout << ss.str();
        //}

        // In case if MatcherPass registered nodes they will be added to the beginning of execution
        // queue
        const auto& new_nodes = m_pass->get_new_nodes(&context);
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
            if (node->inputs().size() != 0) {
                // TODO: exlpore in more details: dequantization operaions are not handled by transformations
                // as result - handle first child after dequantization operation to create new threads
                auto parent = node->get_input_node_shared_ptr(0);
                auto target_inputs = parent->output(0).get_target_inputs();

                //if (parent->get_friendly_name() == "bottleneck3_7/add/fq_input_0") {
                //    std::cout << "TO DEBUG" << std::endl;
                //}

#ifdef DEBUG_THREADING
                const std::set<std::string> toDebug = {
                    // parallelization section #1
                    //"bottleneck2_0/dim_red/conv/fq_input_0",
                    // parallelization section #2
                    //"bottleneck3_0/dim_red/conv/fq_input_0/Multiply",
                    //"bottleneck3_7/add/fq_input_0"
                };

                if (toDebug.find(node->get_friendly_name()) != toDebug.end()) {
                    ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/poc/cpu.current.svg").run_on_function(f);
                    std::cout << "apply_matcher_passes (" << std::this_thread::get_id() << "): " <<
                        parent->get_friendly_name() << " (" << parent->get_type_name() << ") -> " <<
                        node->get_friendly_name() << " (" << node->get_type_name() << ")" <<
                        std::endl;
                }
#endif

                // FIXME: Multiply with the same consumers issue workaround
                bool multiplyWithDifferentConsumers = false;
                if (is_type<opset1::Multiply>(parent)) {
                    multiplyWithDifferentConsumers = false;
                    Node* target_input = nullptr;
                    for (auto it : target_inputs) {
                        if (target_input == nullptr) {
                            target_input = it.get_node();
                        } else {
                            if (target_input->get_friendly_name() != it.get_node()->get_friendly_name()) {
                                multiplyWithDifferentConsumers = true;
                                break;
                            }
                        }
                    }
                }

                if (multiplyWithDifferentConsumers &&
                    ((parent->outputs().size() > 1ul) || (target_inputs.size() > 1ul))) {
                    auto attribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(parent);
                    if (attribute == nullptr) {
                        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/poc/cpu.absent.svg").run_on_function(f);
                        std::stringstream ss;
                        ss << "attribute is absent for node (" << std::this_thread::get_id() << "): " <<
                            parent->get_friendly_name() << " (" << parent->get_type_name() << ")" <<
                            std::endl;
                        std::cout << ss.str();
                    } else if (!attribute->get().handled && (attribute->get().output_thread_ids.size() > 1ul)) {
#ifdef DEBUG_THREADING
                        std::stringstream  ss;
                        ss << "apply_matcher_passes_in_thread (thread: " <<
                            std::this_thread::get_id() << "): " <<
                            parent->get_friendly_name() << " (" << parent->get_type_name() << ")" << std::endl;
                        std::cout << ss.str();
#endif
                        rewritten = rewritten | apply_matcher_passes_in_thread(f, parent);
                        // FIXME: stop iteration
                        return true;
                    }
                }
            }

#ifdef DEBUG_THREADING
            const auto& threadAttribute = ngraph::pass::low_precision::getAttribute<ThreadAttribute>(node);
            if (threadAttribute != nullptr) {
                threadAttribute->get().handled_thread_id = std::this_thread::get_id();
            } else {
                auto& rt = node->get_rt_info();
                auto threadAttribute2 = std::make_shared<ngraph::VariantWrapper<ThreadAttribute>>(ThreadAttribute(0));
                rt[ngraph::VariantWrapper<ThreadAttribute>::type_info.name] = threadAttribute2;
                threadAttribute2->get().handled_thread_id = std::this_thread::get_id();
            }
#endif

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
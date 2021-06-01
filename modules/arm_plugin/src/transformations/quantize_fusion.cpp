// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "quantize_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include <ie_algorithm.hpp>

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "opset/opset.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ArmPlugin;
using Types = std::vector<ngraph::element::Type>;
template<typename NodeIO>
static bool quantized(const NodeIO& node_io) {
    return (node_io.get_element_type() == ngraph::element::u8 ||
            node_io.get_element_type() == ngraph::element::i8);
}

template <class Node>
void ArmPlugin::pass::FakeQuantizeFusionBase::registerMatcher(const std::string& name) {
    auto node_pattern = ngraph::pattern::wrap_type<Node>(ngraph::pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        node_pattern,
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fq_pattern, name),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto node = pattern_map[node_pattern].get_node_shared_ptr();
            auto fakeQuantize = ngraph::as_type_ptr<opset::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
            if (node->output(0).get_target_inputs().size() != 1) {
                return false;
            }
            Types inputTypes;
            std::vector<ngraph::Output<ngraph::Node>> newInputs;
            for (auto&& input : node->inputs()) {
                inputTypes.emplace_back(ngraph::element::f32);
                newInputs.emplace_back(
                    ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), ngraph::element::f32}.get());
            }
            auto fqOutputType = fakeQuantize->get_output_element_type(0);

            auto outputType = fqOutputType;
            auto itMaybeQuantized = node->get_rt_info().find("MayBeQuanitzed");
            if (quantized(node->input(0))) {
                outputType = node->get_input_element_type(0);
            } else if (node->inputs().size() > 1) {
                if (quantized(node->input(1))) {
                    outputType = node->get_input_element_type(1);
                }
            } else if (itMaybeQuantized != node->get_rt_info().end()) {
                auto type = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::element::Type>>(itMaybeQuantized->second);
                IE_ASSERT(type != nullptr);
                outputType = type->get();
            }

            auto new_node = std::make_shared<ngraph::op::TypeRelaxed<Node>>(
                        *std::static_pointer_cast<Node>(node->copy_with_new_inputs(newInputs)),
                        inputTypes,
                        Types{outputType});

            auto quantizationInfo = opset::makeQuantizationInfo(fakeQuantize->input_value(1), fakeQuantize->input_value(2),
                                                                fakeQuantize->input_value(3), fakeQuantize->input_value(4));
            auto isDequantizationSubtractAfterFq = [&] {
                if (fakeQuantize->output(0).get_target_inputs().size() == 1) {
                    auto nodeAfterFq = fakeQuantize->output(0).get_target_inputs().begin()->get_node();
                    if (auto substruct = ngraph::as_type<opset::Subtract>(nodeAfterFq)) {
                        if (substruct->get_friendly_name().find("DequantizationSubtract") != std::string::npos) {
                            return true;
                        }
                    }
                }
                return false;
            } ();

            std::shared_ptr<ngraph::Node> nodeToReplace;
            if (isDequantizationSubtractAfterFq) {
                nodeToReplace = fakeQuantize->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                auto constant = ngraph::as_type<opset::Constant>(nodeToReplace->input_value(1).get_node());
                std::vector<std::int32_t> newOffset;
                auto addOffset = [&] (auto& constOffset) {
                    auto oldOffset = quantizationInfo.offset();
                    auto maxSize = std::max(constOffset.size(), oldOffset.size());
                    constOffset.resize(maxSize, constOffset.back());
                    oldOffset.resize(maxSize, oldOffset.back());
                    for (std::size_t i = 0; i < maxSize; ++i) {
                        newOffset.push_back(oldOffset.at(i) - constOffset.at(i));
                    }
                };
                if (constant->output(0).get_element_type() == ngraph::element::u8) {
                    auto constOffset = constant->cast_vector<std::uint8_t>();
                    addOffset(constOffset);
                } else if (constant->output(0).get_element_type() == ngraph::element::i8) {
                    auto constOffset = constant->cast_vector<std::int8_t>();
                    addOffset(constOffset);
                }
                            // std::vector<float> invScale;
                quantizationInfo = arm_compute::QuantizationInfo{quantizationInfo.scale(), newOffset};
            } else {
                nodeToReplace = fakeQuantize;
            }

            auto fqTargetInputs = fakeQuantize->output(0).get_target_inputs();
            bool fqTargetsMaybeQuantized = std::any_of(std::begin(fqTargetInputs), std::end(fqTargetInputs), [](auto& input) {
                auto itMaybeQuantized = input.get_node()->get_rt_info().find("MayBeQuanitzed");
                return (itMaybeQuantized != input.get_node()->get_rt_info().end());
            });

            auto armNoOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                Types{outputType}, Types{outputType},
                new_node);
            armNoOp->set_friendly_name(new_node->get_friendly_name() + "_arm_noop");
            armNoOp->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    arm_compute::QuantizationInfo{1, 0}));
            if ((!fqTargetsMaybeQuantized) &&
                (fqOutputType == ngraph::element::Type_t::f16 || fqOutputType == ngraph::element::Type_t::f32)) {
                new_node->set_friendly_name(node->get_friendly_name());
                auto armDequantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(
                        Types{outputType}, Types{fqOutputType},
                        armNoOp,
                        fakeQuantize->input_value(1), fakeQuantize->input_value(2),
                        fakeQuantize->input_value(3), fakeQuantize->input_value(4),
                        fakeQuantize->get_levels(), fakeQuantize->get_auto_broadcast());
                armDequantize->set_friendly_name(fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info(fakeQuantize, armDequantize);
                ngraph::replace_node(nodeToReplace, armDequantize);
            } else {
                new_node->set_friendly_name(node->get_friendly_name() + '_' + fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, fakeQuantize}, new_node);
                ngraph::replace_node(nodeToReplace, armNoOp);
            }

            std::vector<float> invScale;
            for (auto&& v : quantizationInfo.scale()) {
                invScale.push_back(v);
            }
            new_node->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    arm_compute::QuantizationInfo{invScale, quantizationInfo.offset()}));

            return true;
        });
}

ArmPlugin::pass::ConvFakeQuantizeFusion::ConvFakeQuantizeFusion() {
    registerMatcher<opset::ArmConvolution>("ArmConvFakeQuantizeFusion");
}

ArmPlugin::pass::GroupConvFakeQuantizeFusion::GroupConvFakeQuantizeFusion() {
    registerMatcher<opset::ArmGroupConvolution>("ArmGroupConvFakeQuantizeFusion");
}

ArmPlugin::pass::MatMulFakeQuantizeFusion::MatMulFakeQuantizeFusion() {
    registerMatcher<opset::MatMul>("AvgMatMulFakeQuantizeFusion");
}

ArmPlugin::pass::AvgPoolFakeQuantizeFusion::AvgPoolFakeQuantizeFusion() {
    registerMatcher<opset::AvgPool>("AvgPoolFakeQuantizeFusion");
}

bool ArmPlugin::pass::PropogateQuantizationInfo::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto&& node : f->get_ordered_ops()) {
        if (!node->inputs().empty()) {
            if (quantized(node->input(0))) {
                auto it_info = node->get_rt_info().find("QuantizationInfo");
                if (it_info == node->get_rt_info().end()) {
                    auto input_it_info = node->get_input_node_ptr(0)->get_rt_info().find("QuantizationInfo");
                    if (input_it_info != node->get_input_node_ptr(0)->get_rt_info().end()) {
                        node->get_rt_info().emplace("QuantizationInfo", input_it_info->second);
                    } else {
                        node->get_rt_info().emplace("QuantizationInfo",
                            std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                            arm_compute::QuantizationInfo{1, 0}));
                    }
                }
            }
        } else if (ngraph::is_type<opset::Constant>(node.get())) {
            if (node->get_output_element_type(0) == ngraph::element::u8 ||
                node->get_output_element_type(0) == ngraph::element::i8) {
                auto it_info = node->get_rt_info().find("QuantizationInfo");
                if (it_info == node->get_rt_info().end()) {
                    node->get_rt_info().emplace("QuantizationInfo",
                                                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                                    arm_compute::QuantizationInfo{1, 0}));
                }
            }
        }
    }
    return false;
}

ArmPlugin::pass::FixQuantizedSubtractOutputDataType::FixQuantizedSubtractOutputDataType() {
    auto sub_pattern = ngraph::pattern::wrap_type<opset::Subtract>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::consumers_count(1));
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(sub_pattern, "FixQuantizedSubtractOutputDataType"),
        [=](ngraph::pattern::Matcher& m) {
            auto sub = std::dynamic_pointer_cast<opset::Subtract>(m.get_match_root());

            if (!(quantized(sub->input(0)) && quantized(sub->input(1)))) {
                return false;
            }

            auto new_sub = std::make_shared<ngraph::op::TypeRelaxed<opset::Subtract>>(
                        Types{sub->get_input_element_type(0), sub->get_input_element_type(1)},
                        Types{sub->get_input_element_type(0)},
                        sub->input_value(0),
                        sub->input_value(1));
            ngraph::copy_runtime_info(sub, new_sub);
            new_sub->set_friendly_name(sub->get_friendly_name());
            ngraph::replace_node(sub, new_sub);
            return true;
        });
}

ArmPlugin::pass::FixMulInputConvert::FixMulInputConvert() {
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::consumers_count(1));
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "FixMulInputConvert"),
        [=](ngraph::pattern::Matcher& m) {
            auto mul = ngraph::as_type_ptr<opset::Multiply>(m.get_match_root());

            if (!((quantized(mul->input(0)) || quantized(mul->input(1))) && !quantized(mul->output(0)))) {
                return false;
            }

            auto outputType = mul->get_output_element_type(0);

            auto new_mul = std::make_shared<ngraph::op::TypeRelaxed<opset::Multiply>>(
                        Types{outputType, outputType},
                        Types{outputType},
                        std::make_shared<opset::Convert>(mul->input_value(0), outputType),
                        std::make_shared<opset::Convert>(mul->input_value(1), outputType));
            ngraph::copy_runtime_info(mul, new_mul);
            new_mul->set_friendly_name(mul->get_friendly_name());
            ngraph::replace_node(mul, new_mul);
            return true;
        });
}

bool ArmPlugin::pass::DetectMaybeQuantized::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto reversedOps = f->get_ordered_ops();
    std::reverse(reversedOps.begin(), reversedOps.end());

    for (auto&& node : reversedOps) {
        auto nodeInput = node->inputs();
        auto nodeOutputs = node->outputs();
        ngraph::element::Type type;
        for (auto&& input : nodeInput) {
            if (quantized(input)) {
                type = input.get_element_type();
                break;
            }
        }
        if (type == ngraph::element::undefined) {
            for (auto&& output : nodeOutputs) {
                if (quantized(output)) {
                    type = output.get_element_type();
                    break;
                }
                auto targetInputs = output.get_target_inputs();
                for (auto&& targetInput : targetInputs) {
                    auto itMaybeQuantized = targetInput.get_node()->get_rt_info().find("MayBeQuanitzed");
                    if (itMaybeQuantized != targetInput.get_node()->get_rt_info().end()) {
                        auto targetType = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::element::Type>>(itMaybeQuantized->second);
                        IE_ASSERT(targetType != nullptr);
                        if (targetType->get() != ngraph::element::undefined) {
                            type = targetType->get();
                            break;
                        }
                    }
                }
                if (type != ngraph::element::undefined) break;
            }
        }
        if (type != ngraph::element::undefined) {
            node->get_rt_info().emplace("MayBeQuanitzed",
                std::make_shared<ngraph::VariantWrapper<ngraph::element::Type>>(type));
        }
    }
    return false;
}

namespace ngraph {
NGRAPH_RTTI_DEFINITION(VariantWrapper<ngraph::element::Type>, "Variant::ngraph::element::Type", 0);
VariantWrapper<ngraph::element::Type>::~VariantWrapper() {}
}  // namespace ngraph
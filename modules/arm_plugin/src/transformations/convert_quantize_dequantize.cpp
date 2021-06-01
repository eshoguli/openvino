// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>
#include "opset/opset.hpp"
#include "transformations/convert_quantize_dequantize.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ArmPlugin::pass::ConvertQuantizeDequantize::ConvertQuantizeDequantize() {
    auto fakeQuantize = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(
        std::make_shared<ngraph::pattern::Matcher>(fakeQuantize, "ConvertArmQuantizeDequantize"),
        [](ngraph::pattern::Matcher& m) {
            auto fakeQuantize = std::dynamic_pointer_cast<opset::FakeQuantize>(m.get_match_root());
            IE_ASSERT(fakeQuantize != nullptr);
            auto input_type = fakeQuantize->input(0).get_element_type();
            auto output_type = fakeQuantize->output(0).get_element_type();
            auto input = fakeQuantize->input_value(0);
            auto input_low = fakeQuantize->input_value(1);
            auto input_high = fakeQuantize->input_value(2);
            auto output_low = fakeQuantize->input_value(3);
            auto output_high = fakeQuantize->input_value(4);
            using Types = std::vector<ngraph::element::Type>;
            if ((input_type == ngraph::element::Type_t::f16 || input_type == ngraph::element::Type_t::f32 ||
                 input_type == ngraph::element::Type_t::i8 || input_type == ngraph::element::Type_t::u8) &&
                (output_type == ngraph::element::Type_t::i8 || output_type == ngraph::element::Type_t::u8)) {
                auto quantizationInfo = opset::makeQuantizationInfo(input_low, input_high, output_low, output_high);
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
                } else {
                    nodeToReplace = fakeQuantize;
                }
                auto armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(
                    Types{input_type}, Types{output_type},
                    input, input_low, input_high, output_low, output_high, fakeQuantize->get_levels(), fakeQuantize->get_auto_broadcast());
                armQuantize->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize");
                ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                armQuantize->get_rt_info().emplace("QuantizationInfo",
                    std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(quantizationInfo));

                auto armNoOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                    Types{output_type}, Types{output_type},
                    armQuantize);
                armNoOp->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_noop");
                armNoOp->get_rt_info().emplace("QuantizationInfo",
                    std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                        arm_compute::QuantizationInfo{1, 0}));
                ngraph::replace_node(nodeToReplace, armNoOp);
                return true;
            } else if ((input_type == ngraph::element::Type_t::i8 || input_type == ngraph::element::Type_t::u8) &&
                       (output_type == ngraph::element::Type_t::f16 || output_type == ngraph::element::Type_t::f32)) {
                auto armDequantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(
                    Types{input_type}, Types{output_type},
                    input, input_low, input_high, output_low, output_high, fakeQuantize->get_levels(), fakeQuantize->get_auto_broadcast());
                armDequantize->set_friendly_name(fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info(fakeQuantize, armDequantize);
                ngraph::replace_node(fakeQuantize, armDequantize);
                return true;
            }
            return false;
        });
}

ArmPlugin::pass::StoreWeightsQuantizationInfo::StoreWeightsQuantizationInfo() {
    auto constant_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto convert_pattern = ngraph::pattern::wrap_type<opset::Convert>({constant_pattern}, ngraph::pattern::consumers_count(1));
    auto zero_point_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto sub_pattern = ngraph::pattern::wrap_type<opset::Subtract>({convert_pattern, zero_point_pattern}, ngraph::pattern::consumers_count(1));
    auto scale_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>({sub_pattern, scale_pattern}, ngraph::pattern::consumers_count(1));
    register_matcher(
        std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "StoreWeightsQuantizationInfo"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto constant = ngraph::as_type<opset::Constant>(pattern_map[constant_pattern].get_node());
            IE_ASSERT(constant != nullptr);
            auto zero_point = ngraph::as_type<opset::Constant>(pattern_map[zero_point_pattern].get_node());
            IE_ASSERT(zero_point != nullptr);
            auto scale = ngraph::as_type<opset::Constant>(pattern_map[scale_pattern].get_node());
            IE_ASSERT(scale != nullptr);
            std::vector<float> scale_vector;
            std::vector<std::int32_t> zero_point_vector;
            if (zero_point->output(0).get_element_type() == ngraph::element::f32) {
                for (auto v : zero_point->cast_vector<float>()) {
                    zero_point_vector.push_back(v);
                }
                for (auto v : scale->cast_vector<float>()) {
                    scale_vector.push_back(v);
                }
            } else if (zero_point->output(0).get_element_type() == ngraph::element::f16) {
                for (auto v : zero_point->cast_vector<ngraph::float16>()) {
                    zero_point_vector.push_back(v);
                }
                for (auto v : scale->cast_vector<ngraph::float16>()) {
                    scale_vector.push_back(v);
                }
            } else {
                IE_THROW() << "Arm Plugin: Unsupported Data type: " << zero_point->output(0).get_element_type();
            }
            constant->get_rt_info()["QuantizationInfo"]
                = std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(arm_compute::QuantizationInfo{
                    scale_vector, zero_point_vector});
            return false;
        });
}

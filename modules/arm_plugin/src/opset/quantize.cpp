// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "quantize.hpp"
#include <half/half.hpp>

namespace ngraph {
NGRAPH_RTTI_DEFINITION(VariantWrapper<arm_compute::QuantizationInfo>, "Variant::arm_compute::QuantizationInfo", 0);
VariantWrapper<arm_compute::QuantizationInfo>::~VariantWrapper() {}
}  // namespace ngraph

using namespace ArmPlugin;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(opset::ArmQuantize, "ArmQuantize", 0);

arm_compute::QuantizationInfo ArmPlugin::opset::makeQuantizationInfo(
                const ngraph::Output<ngraph::Node>& input_low_output,
                const ngraph::Output<ngraph::Node>& input_high_output,
                const ngraph::Output<ngraph::Node>& output_low_output,
                const ngraph::Output<ngraph::Node>& output_high_output) {
    auto data_type = input_low_output.get_element_type();
    std::vector<float> scale_vector;
    std::vector<std::int32_t> zero_point_vector;
    auto add_chanel = [&](float min, float max, float qMin, float qMax) {
        auto scale = (max - min) / (qMax - qMin);
        auto zeroPointReal = qMin - min / scale;
        std::int32_t zeroPointNudged = 0;
        if (zeroPointReal < qMin) {
            zeroPointNudged = qMin;
        } else if (zeroPointReal > qMax) {
            zeroPointNudged = qMax;
        } else {
            zeroPointNudged = static_cast<std::int32_t>(std::round(zeroPointReal));
        }
        scale_vector.emplace_back(scale);
        zero_point_vector.emplace_back(zeroPointNudged);
    };
    auto init = [&] (auto get_vector) {
        auto input_low = get_vector(input_low_output);
        auto input_high = get_vector(input_high_output);
        auto output_low = get_vector(output_low_output);
        auto output_high = get_vector(output_high_output);
        IE_ASSERT(input_low.size() == input_high.size());
        for (std::size_t i = 0; i < input_low.size(); ++i) {
            add_chanel(input_low[i], input_high[i], output_low[0], output_high[0]);
        }
    };
    if (data_type == ngraph::element::Type_t::f16) {
        init([&](const ngraph::Output<ngraph::Node>& input) {
            return ngraph::as_type<opset::Constant>(input.get_node())->cast_vector<ngraph::float16>();
        });
    } else if (data_type == ngraph::element::Type_t::f32) {
        init([&](const ngraph::Output<ngraph::Node>& input) {
            return ngraph::as_type<opset::Constant>(input.get_node())->cast_vector<float>();
        });
    } else {
        IE_THROW() << "Arm Plugin: Unsupported Data type: " << data_type;
    }
    return {scale_vector, zero_point_vector};
}


opset::ArmQuantize::ArmQuantize(const ngraph::Output<ngraph::Node>& data,
                                const ngraph::Output<ngraph::Node>& input_low,
                                const ngraph::Output<ngraph::Node>& input_high,
                                const ngraph::Output<ngraph::Node>& output_low,
                                const ngraph::Output<ngraph::Node>& output_high,
                                std::size_t levels,
                                const ngraph::op::AutoBroadcastSpec& auto_broadcast) :
    FakeQuantize{data, input_low, input_high, output_low, output_high, levels, auto_broadcast} {}

opset::ArmQuantize::~ArmQuantize() {}

std::shared_ptr<Node> opset::ArmQuantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmQuantize>(new_args.at(0), // X
                                         new_args.at(1), // input_low
                                         new_args.at(2), // input_high
                                         new_args.at(3), // output_low
                                         new_args.at(4), // output_high
                                         get_levels(),
                                         get_auto_broadcast());
}

NGRAPH_RTTI_DEFINITION(opset::ArmDequantize, "ArmDequantize", 0);

opset::ArmDequantize::ArmDequantize(const ngraph::Output<ngraph::Node>& data,
                                    const ngraph::Output<ngraph::Node>& input_low,
                                    const ngraph::Output<ngraph::Node>& input_high,
                                    const ngraph::Output<ngraph::Node>& output_low,
                                    const ngraph::Output<ngraph::Node>& output_high,
                                    std::size_t levels,
                                    const ngraph::op::AutoBroadcastSpec& auto_broadcast) :
    FakeQuantize{data, input_low, input_high, output_low, output_high, levels, auto_broadcast} {}

opset::ArmDequantize::~ArmDequantize() {}

std::shared_ptr<Node> opset::ArmDequantize::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ArmDequantize>(new_args.at(0), // X
                                           new_args.at(1), // input_low
                                           new_args.at(2), // input_high
                                           new_args.at(3), // output_low
                                           new_args.at(4), // output_high
                                           get_levels(),
                                           get_auto_broadcast());
}

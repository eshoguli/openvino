// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/variant.hpp>
#include "ngraph/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"
#include "arm_compute/core/QuantizationInfo.h"

namespace ArmPlugin {
namespace opset {

arm_compute::QuantizationInfo makeQuantizationInfo(
                const ngraph::Output<ngraph::Node>& input_low,
                const ngraph::Output<ngraph::Node>& input_high,
                const ngraph::Output<ngraph::Node>& output_low,
                const ngraph::Output<ngraph::Node>& output_high);

struct ArmQuantize : public FakeQuantize {
    NGRAPH_RTTI_DECLARATION;
    ArmQuantize(const ngraph::Output<ngraph::Node>& data,
                const ngraph::Output<ngraph::Node>& input_low,
                const ngraph::Output<ngraph::Node>& input_high,
                const ngraph::Output<ngraph::Node>& output_low,
                const ngraph::Output<ngraph::Node>& output_high,
                std::size_t levels,
                const ngraph::op::AutoBroadcastSpec& auto_broadcast = {ngraph::op::AutoBroadcastType::NUMPY});
    ~ArmQuantize() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};

struct ArmDequantize : public FakeQuantize {
    NGRAPH_RTTI_DECLARATION;
    ArmDequantize(const ngraph::Output<ngraph::Node>& data,
                  const ngraph::Output<ngraph::Node>& input_low,
                  const ngraph::Output<ngraph::Node>& input_high,
                  const ngraph::Output<ngraph::Node>& output_low,
                  const ngraph::Output<ngraph::Node>& output_high,
                  std::size_t levels,
                  const ngraph::op::AutoBroadcastSpec& auto_broadcast = {ngraph::op::AutoBroadcastType::NUMPY});
    ~ArmDequantize() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
};

}  // namespace opset
}  // namespace ArmPlugin

namespace ngraph {
template <>
struct NGRAPH_API VariantWrapper<arm_compute::QuantizationInfo> : public VariantImpl<arm_compute::QuantizationInfo> {
    NGRAPH_RTTI_DECLARATION;
    VariantWrapper(const arm_compute::QuantizationInfo& value) : VariantImpl<arm_compute::QuantizationInfo>{value} {}
    ~VariantWrapper() override;
};
}  // namespace ngraph

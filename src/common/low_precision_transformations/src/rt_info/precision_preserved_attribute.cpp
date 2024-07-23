// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precision_preserved_attribute.hpp"

#include <memory>
#include <string>

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset12.hpp"

#include "low_precision/network_helper.hpp"

using namespace ov;
using namespace ov;

PrecisionPreservedAttribute::PrecisionPreservedAttribute(const bool value) :
    SharedAttribute(value) {
}

std::string PrecisionPreservedAttribute::to_string() const {
    std::stringstream ss;
    ss << attribute->get_string();
    ss << "value: " << (value() ? "true" : "false");
    return ss.str();
}

//bool PrecisionPreservedAttribute::is_copyable() const {
//    return false;
//}
//
//bool PrecisionPreservedAttribute::is_copyable(const std::shared_ptr<Node>& to) const {
//    return false;
//}

namespace {
template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool isDisabled(const std::shared_ptr<Node>& node) {
    for (const auto& input : node->inputs()) {
        auto precisionAttribute = ov::pass::low_precision::getAttribute<PrecisionsAttribute>(input);
        if (precisionAttribute.empty()) {
            continue;
        }
        const auto& precisionRestrictions = precisionAttribute.as<PrecisionsAttribute>().value();
        if (precisionRestrictions.empty()) {
            return true;
        }
    }
    return false;
}
} // namespace

bool PrecisionPreservedAttribute::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
    if (isDisabled(node)) {
        return false;
    }

    using namespace ov;
    // TODO: think how to handle conditions <= not mandatory for PoC
    // TODO: operation set version is not affected <= not mandatory for PoC
    static const std::unordered_set<std::string> precisionPreservedOps = {
        { name<opset1::Concat>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::Interpolate>() },
        { name<opset1::MaxPool>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset2::BatchToSpace>() },
        { name<opset1::Broadcast>() },
        { name<opset1::Pad>() },
        { name<opset12::Pad>() },
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset2::SpaceToBatch>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() }
    };

    if (precisionPreservedOps.find(node->get_type_name()) != precisionPreservedOps.end()) {
        return true;
    }

    if (ov::is_type<opset1::Interpolate>(node)) {
        std::shared_ptr<opset1::Interpolate> interpolate1 = ov::as_type_ptr<opset1::Interpolate>(node);
        if (interpolate1) {
            const auto attrs = interpolate1->get_attrs();
            return attrs.mode == "nearest";
        }

        std::shared_ptr<opset4::Interpolate> interpolate4 = ov::as_type_ptr<opset4::Interpolate>(node);
        if (interpolate4) {
            const auto attrs = interpolate4->get_attrs();
            return attrs.mode == op::v4::Interpolate::InterpolateMode::NEAREST;
        }
    }

    return false;
}
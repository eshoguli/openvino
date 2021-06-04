// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <low_precision/common/subgraph_state.hpp>
#include "low_precision/quantization_details.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"


namespace ngraph {
namespace pass {
namespace low_precision {

SubgraphState::SubgraphState(const Subgraph& subgraph) {
    for (const auto fq : subgraph.quantizationLayers) {
        original.emplace(fq->get_friendly_name(), QuantizationDetails::getDetails(ngraph::as_type_ptr<opset1::FakeQuantize>(fq)));
    }
}

namespace {

void compareDequantization(
    const std::string& friendlyName,
    const QuantizationDetails& original,
    const QuantizationDetails& result) {
    if (original.levels != result.levels) {
        std::cout << "compareDequantization: " << friendlyName << ": levels values are different" <<
            original.levels << " VS " << result.levels << std::endl;
    }

    if (original.outputLowValues.size() != result.outputLowValues.size()) {
        std::cout << "compareDequantization: " << friendlyName << ": output low boundaries count is different" <<
            original.outputLowValues.size() << " VS " << result.outputLowValues.size() << std::endl;
    }

    if (original.outputHighValues.size() != result.outputHighValues.size()) {
        std::cout << "compareDequantization: " << friendlyName << ": output high boundaries count is different" <<
            original.outputHighValues.size() << " VS " << result.outputHighValues.size() << std::endl;
    }

    for (size_t i = 0ul; i < original.outputLowValues.size(); ++i) {
        if (original.outputLowValues[i] != result.outputLowValues[i]) {
            std::cout << "compareDequantization: " << friendlyName << ": output low [" << i << "] values are not equals: " <<
                original.outputLowValues[i] << " VS " << result.outputLowValues[i] << std::endl;
        }
    }

    for (size_t i = 0ul; i < original.outputHighValues.size(); ++i) {
        if (original.outputHighValues[i] != result.outputHighValues[i]) {
            std::cout << "compareDequantization: " << friendlyName << ": output high [" << i << "] values are not equals: " <<
                original.outputHighValues[i] << " VS " << result.outputHighValues[i] << std::endl;
        }
    }
}

void checkAlignment(
    const std::string& friendlyName,
    const QuantizationDetails& original,
    const QuantizationDetails& result) {
    if (original.levels != result.levels) {
        std::cout << "checkAlignment: " << friendlyName << ": levels values are different: " <<
            original.levels << " VS " << result.levels << std::endl;
    }
}

} // namespace

void SubgraphState::compare(std::shared_ptr<ngraph::Function>& function) {
    for (const auto op : function->get_ops()) {
        const auto it = original.find(op->get_friendly_name());
        if (it != original.end()) {
            const QuantizationDetails result = QuantizationDetails::getDetails(ngraph::as_type_ptr<opset1::FakeQuantize>(op));
            const QuantizationDetails original = it->second;
            checkAlignment(op->get_friendly_name(), original, result);
        }
    }
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

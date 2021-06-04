// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/check.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "../ilayer_transformations_manager.hpp"
#include <low_precision/common/subgraph.hpp>
#include <low_precision/quantization_details.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class SubgraphState {
public:
    SubgraphState(const Subgraph& subgraph);
    void compare(std::shared_ptr<ngraph::Function>& function);

private:
    std::map<std::string, QuantizationDetails> original;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph

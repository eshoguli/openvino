// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class UpdatePrecision : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("UpdatePrecision", "0");
    UpdatePrecision(const ngraph::element::Type keep_precision);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

private:
    element::Type keep_precision;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
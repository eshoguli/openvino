// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/variant.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {
struct ConvertQuantizeDequantize: public ngraph::pass::MatcherPass {
    ConvertQuantizeDequantize();
};

struct StoreWeightsQuantizationInfo: public ngraph::pass::MatcherPass {
    StoreWeightsQuantizationInfo();
};
}  // namespace pass
}  // namespace ArmPlugin

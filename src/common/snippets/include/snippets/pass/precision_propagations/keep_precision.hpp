// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>
#include <snippets/itt.hpp>
#include "snippets/op/convert_saturation.hpp"

#include "snippets/pass/precision_propagations/default_pass.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagations {

class KeepPrecision: public DefaultPass {
public:
    OPENVINO_RTTI("KeepPrecision", "0");
    KeepPrecision(const ov::element::Type precision);
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/constant.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void ConstantTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::Constant>(pass, context);
}

bool ConstantTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(m.get_match_root());
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto values = constant->cast_vector<float>();
    auto newValues = std::vector<float>(values.size());

    bool wasChangedZero = false;
    bool wasChangedDenormal = false;
    for (size_t i = 0; i < values.size(); i++) {
        const auto value = values[i];
        const auto abs_value = std::abs(static_cast<double>(value));

        //if (abs_value == 0.f) {
        //    wasChangedZero = true;
        //    newValues[i] = 1.f;
        //} else
        if ((abs_value > 0.f) && (abs_value < 1.e-32)) {
            wasChangedDenormal = true;
            newValues[i] = 0.f;
        } else if (abs_value > 1.e+32) {
            wasChangedDenormal = true;
            newValues[i] = 999.f;
        } else {
            newValues[i] = value;
        }
    }

    if (!wasChangedZero && !wasChangedDenormal) {
        return false;
    }

    const auto newConstant = std::make_shared<opset1::Constant>(
        constant->output(0).get_element_type(),
        constant->output(0).get_shape(),
        newValues);
    newConstant->set_friendly_name(constant->get_friendly_name());
    replace_node(constant, newConstant);

    std::cout << "ConstantTransformation::transform: " << (wasChangedDenormal ? "denormal: " : "zero: ") << constant->get_friendly_name() << std::endl;
    return true;
}

bool ConstantTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return true;
}

bool ConstantTransformation::isPrecisionPreserved(std::shared_ptr<ngraph::Node> node) const noexcept {
    return false;
}

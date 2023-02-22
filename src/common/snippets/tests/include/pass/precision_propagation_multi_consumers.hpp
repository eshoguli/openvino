// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationMultiConsumerParamsValues {
public:
    class Actual {
    public:
        Actual() = default;

        Actual(
            const std::set<std::vector<element::Type>>& supported_precisions_maximum,
            const std::set<std::vector<element::Type>>& supported_precisions_minimum,
            const std::set<std::vector<element::Type>>& supported_precisions_add,
            const element::Type convertion_after_parameter1_all = {},
            const element::Type convertion_after_parameter1_min_max = {},
            const element::Type convertion_after_parameter2 = {},
            const element::Type convertion_before_maximum = {},
            const element::Type convertion_after_maximum = {},
            const element::Type convertion_before_minimum = {},
            const element::Type convertion_after_minimum = {},
            const element::Type convertion_before_add = {},
            const element::Type convertion_after_add = {}) :
            supported_precisions_maximum(supported_precisions_maximum),
            supported_precisions_minimum(supported_precisions_minimum),
            supported_precisions_add(supported_precisions_add),
            convertion_after_parameter1_all(convertion_after_parameter1_all),
            convertion_after_parameter1_min_max(convertion_after_parameter1_min_max),
            convertion_after_parameter2(convertion_after_parameter2),
            convertion_before_maximum(convertion_before_maximum),
            convertion_after_maximum(convertion_after_maximum),
            convertion_before_minimum(convertion_before_minimum),
            convertion_after_minimum(convertion_after_minimum),
            convertion_before_add(convertion_before_add),
            convertion_after_add(convertion_after_add) {}

        std::set<std::vector<element::Type>> supported_precisions_maximum;
        std::set<std::vector<element::Type>> supported_precisions_minimum;
        std::set<std::vector<element::Type>> supported_precisions_add;
        element::Type convertion_after_parameter1_all;
        element::Type convertion_after_parameter1_min_max;
        element::Type convertion_after_parameter2;
        element::Type convertion_before_maximum;
        element::Type convertion_after_maximum;
        element::Type convertion_before_minimum;
        element::Type convertion_after_minimum;
        element::Type convertion_before_add;
        element::Type convertion_after_add;
    };

    class Expected {
    public:
        //Expected() = default;

        Expected(
            const element::Type convertion_after_parameter1_all = {},
            const element::Type convertion_after_parameter1_min_max = {},
            const element::Type convertion_after_parameter2 = {},
            const element::Type convertion_before_maximum = {},
            const element::Type convertion_after_maximum = {},
            const element::Type convertion_before_minimum = {},
            const element::Type convertion_after_minimum = {},
            const element::Type convertion_before_add = {},
            const element::Type convertion_after_add = {}) :
            convertion_after_parameter1_all(convertion_after_parameter1_all),
            convertion_after_parameter1_min_max(convertion_after_parameter1_min_max),
            convertion_after_parameter2(convertion_after_parameter2),
            convertion_before_maximum(convertion_before_maximum),
            convertion_after_maximum(convertion_after_maximum),
            convertion_before_minimum(convertion_before_minimum),
            convertion_after_minimum(convertion_after_minimum),
            convertion_before_add(convertion_before_add),
            convertion_after_add(convertion_after_add) {}

        element::Type convertion_after_parameter1_all;
        element::Type convertion_after_parameter1_min_max;
        element::Type convertion_after_parameter2;
        element::Type convertion_before_maximum;
        element::Type convertion_after_maximum;
        element::Type convertion_before_minimum;
        element::Type convertion_after_minimum;
        element::Type convertion_before_add;
        element::Type convertion_after_add;
    };

    PrecisionPropagationMultiConsumerParamsValues() = default;

    PrecisionPropagationMultiConsumerParamsValues(
        const std::vector<element::Type>& input_types,
        const Actual actual,
        const Expected expected) : input_types(input_types), actual(actual), expected(expected) {}

    std::vector<element::Type> input_types;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<Shape, Shape>, // input shapes
    PrecisionPropagationMultiConsumerParamsValues
> PrecisionPropagationMultiConsumerParams;

class PrecisionPropagationMultiConsumerTest : public TransformationTestsF,
                                 public testing::WithParamInterface<PrecisionPropagationMultiConsumerParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PrecisionPropagationMultiConsumerParams> obj);

protected:
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/enforce_precision.hpp"
#include <gtest/gtest.h>
#include <ngraph/ngraph.hpp>

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> input_shapes = {
    {{ 1, 16, 384, 64 }, { 1, 16, 64, 384 }},
};

namespace platform_bf16 {
const std::vector<EnforcePrecisionTestValues> test_values_bf16 = {
    {
        false,
        true
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_EnforcePrecision_bf16, EnforcePrecisionTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(test_values_bf16),
                            ::testing::Values(7),
                            ::testing::Values(1),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        EnforcePrecisionTest::getTestCaseName);
} // platform_bf16

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov

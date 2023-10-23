// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/eltwise.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Eltwise {
namespace {
static const std::vector<InputShape> bitwise_in_shapes_4D = {
    {
        {1, -1, -1, -1},
        {
            {1, 3, 4, 4},
            {1, 3, 1, 1},
            {1, 1, 1, 1}
        }
    },
    {{1, 3, 4, 4}, {{1, 3, 4, 4}}}
};

const auto params_4D_bitwise = ::testing::Combine(
    ::testing::Combine(
        ::testing::Values(bitwise_in_shapes_4D),
        ::testing::ValuesIn({
            ngraph::helpers::EltwiseTypes::BITWISE_AND,
            ngraph::helpers::EltwiseTypes::BITWISE_OR,
            ngraph::helpers::EltwiseTypes::BITWISE_XOR
        }),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ nhwc, nhwc }, { nhwc }, {}, "ref")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise, EltwiseLayerCPUTest, params_4D_bitwise, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_i16 = ::testing::Combine(
    ::testing::Combine(
        ::testing::Values(bitwise_in_shapes_4D),
        ::testing::ValuesIn({
            ngraph::helpers::EltwiseTypes::BITWISE_AND,
            ngraph::helpers::EltwiseTypes::BITWISE_OR,
            ngraph::helpers::EltwiseTypes::BITWISE_XOR
            }),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i16, ov::element::Type_t::u16 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ nhwc, nhwc }, { nhwc }, {}, "ref_I32$/")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_i16, EltwiseLayerCPUTest, params_4D_bitwise_i16, EltwiseLayerCPUTest::getTestCaseName);


const auto params_4D_bitwise_NOT = ::testing::Combine(
    ::testing::Combine(
        ::testing::Values(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ ngraph::helpers::EltwiseTypes::BITWISE_NOT }),
        ::testing::ValuesIn({ ngraph::helpers::InputLayerType::NONE }),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i8, /*ov::element::Type_t::u8,*/ ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ nhwc }, { nhwc }, {}, "ref")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT, EltwiseLayerCPUTest, params_4D_bitwise_NOT, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_NOT_i16 = ::testing::Combine(
    ::testing::Combine(
        ::testing::Values(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ ngraph::helpers::EltwiseTypes::BITWISE_NOT }),
        ::testing::ValuesIn({ ngraph::helpers::InputLayerType::NONE }),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i16, ov::element::Type_t::u16 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ nhwc }, { nhwc }, {}, "ref_I32$/")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT_i16, EltwiseLayerCPUTest, params_4D_bitwise_NOT, EltwiseLayerCPUTest::getTestCaseName);

} // namespace
} // namespace Eltwise
} // namespace CPULayerTestsDefinitions

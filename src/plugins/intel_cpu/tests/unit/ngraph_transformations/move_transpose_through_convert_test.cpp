// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <tuple>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"
#include "snippets_transformations/move_transpose_through_convert.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_convert_transpose_convert.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace move_transpose_through_convert_test {

class MoveTransposeThroughConvertParamsValues {
public:
    class Original {
    public:
        element::Type convert_before_precision;
        element::Type convert_after_precision;
    };
    class Expected {
    public:
        element::Type convert_before_precision;
        element::Type convert_after_precision1;
        element::Type convert_after_precision2;
    };

    const ngraph::element::Type input_precision;
    const std::vector<size_t>& transpose_order;
    Original original;
    Expected expected;
};

typedef std::tuple<
    PartialShape, // input shapes
    MoveTransposeThroughConvertParamsValues
> MoveTransposeThroughConvertParams;

class MoveTransposeThroughConvertTest : public TransformationTestsF,
                                        public testing::WithParamInterface<MoveTransposeThroughConvertParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MoveTransposeThroughConvertParams> obj) {
        const auto shapes = std::get<0>(obj.param);
        const auto test_values = std::get<1>(obj.param);

        auto to_string = [](const std::set<std::vector<element::Type>>& precisions_pack) noexcept {
            std::ostringstream result;
            result << "{";
            for (const auto& precisions : precisions_pack) {
                result << CommonTestUtils::vec2str(precisions) << "_";
            }
            result << "}";
            return result.str();
        };

        std::ostringstream result;
        result << "in=" << shapes << "_"
               << "original_before=" << test_values.original.convert_before_precision << "_"
               << "original_after=" << test_values.original.convert_after_precision << "_"
               << "expected_before=" << test_values.expected.convert_before_precision << "_"
               << "expected_after1=" << test_values.expected.convert_after_precision1 << "_"
               << "expected_after2=" << test_values.expected.convert_after_precision2;
        return result.str();
    }
};

TEST_P(MoveTransposeThroughConvertTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    const auto input_shapes = std::vector<PartialShape>({ shapes });
    SubgraphConvertTransposeConvertFunction function_stub(
        input_shapes,
        test_values.input_precision,
        test_values.transpose_order,
        {
            test_values.original.convert_before_precision,
            test_values.original.convert_after_precision
        },
        {
            test_values.expected.convert_before_precision,
            test_values.expected.convert_after_precision1,
            test_values.expected.convert_after_precision2
        });
    function = function_stub.getOriginal();

    ngraph::pass::VisualizeTree("svg/test.original.svg").run_on_model(function);

    // TODO: for tests only: use base type manager
    ngraph::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::pass::MoveTransposeThroughConvert>(element::f32, element::bf16);
    manager.run_passes(function);

    ngraph::pass::VisualizeTree("svg/test.transformed.svg").run_on_model(function);

    function_ref = function_stub.getReference();

    ngraph::pass::VisualizeTree("svg/test.reference.svg").run_on_model(function_ref);
}

// clang-format off

std::vector<PartialShape> shapes {
    {1, 3, 16, 16}
};

std::vector<MoveTransposeThroughConvertParamsValues> test_cases {
    {
        element::bf16,
        {0, 2, 3, 1}, // supported
        {
            {element::f32},
            {element::bf16}
        },
        {
            {},
            {element::f32},
            {element::bf16}
        }
    },
    {
        element::bf16,
        {0, 2, 1, 3}, // not supported
        {
            {element::f32},
            {element::bf16}
        },
        {
            {element::f32},
            {element::bf16},
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MoveTransposeThroughConvertTest,
    MoveTransposeThroughConvertTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(test_cases)),
    MoveTransposeThroughConvertTest::getTestCaseName);

// clang-format on
} // namespace move_transpose_through_convert_test

}  // namespace snippets
}  // namespace test
}  // namespace ov

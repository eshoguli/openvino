// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "snippets_transformations/enforce_precision.hpp"
#include "common_test_utils/common_utils.hpp"
#include "two_element_wise_function.hpp"
#include "lowering_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

class DummyPrecisionSelection {
public:
    DummyPrecisionSelection(
        const ov::intel_cpu::pass::EnforcePrecision::Operation& op1,
        const ov::intel_cpu::pass::EnforcePrecision::Operation& op2) : op1(op1), op2(op2) {
    }

    ov::intel_cpu::pass::EnforcePrecision::Operation get_supported_precisions(const std::shared_ptr<ngraph::Node>& op) noexcept {
        if (ov::is_type<ov::test::snippets::DummyOperation1>(op)) {
            return op1;
        } else if (ov::is_type<ov::test::snippets::DummyOperation2>(op)) {
            return op2;
        }
        return {};
    }

private:
    const ov::intel_cpu::pass::EnforcePrecision::Operation op1;
    const ov::intel_cpu::pass::EnforcePrecision::Operation op2;
};

} // namespace

class EnforcePrecisionParamsValues {
public:
    class Actual {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
        ov::intel_cpu::pass::EnforcePrecision::Operation op1;
        ov::intel_cpu::pass::EnforcePrecision::Operation op2;
    };

    class Expected {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
        element::Type convertion_before_result;
    };

    std::vector<element::Type> input_types;
    bool bf16_isa;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<PartialShape, PartialShape>, // input shapes
    EnforcePrecisionParamsValues
> EnforcePrecisionParams;

class EnforcePrecisionTest : public TransformationTestsF,
                             public testing::WithParamInterface<EnforcePrecisionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EnforcePrecisionParams> obj) {
        std::pair<PartialShape, PartialShape> shapes;
        EnforcePrecisionParamsValues test_values;
        std::tie(shapes, test_values) = obj.param;

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
        result << "IN0=" << shapes.first << "_" << test_values.input_types[0] << "_"
            << "IN1=" << shapes.second << "_" << test_values.input_types[1] << "_"
            << "IN2=" << test_values.input_types[2] << "_"
            << "bf16_isa=" << test_values.bf16_isa << "_"
            << test_values.actual.op1.only_after_parameter << "_" << to_string(test_values.actual.op1.precisions) << "_"
            << test_values.actual.op2.only_after_parameter << "_" << to_string(test_values.actual.op2.precisions) << "_"
            << test_values.expected.convertion_before_op1.first << "_" << test_values.expected.convertion_before_op1.second << "_"
            << test_values.expected.convertion_before_op2_1 << "_"
            << test_values.expected.convertion_before_op2_2.first << "_" << test_values.expected.convertion_before_op2_2.second << "_"
            << test_values.expected.convertion_after_op2 << "_";
        return result.str();
    }
};

TEST_P(EnforcePrecisionTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    const auto input_shapes = std::vector<PartialShape>({ shapes.first, shapes.second });
    TwoElementWiseFunction function_stub(
        input_shapes,
        test_values.input_types[0],
        test_values.input_types[1],
        test_values.input_types[2],
        {
            test_values.actual.convertion_before_op1,
            test_values.actual.convertion_before_op2_1,
            test_values.actual.convertion_before_op2_2,
            test_values.actual.convertion_after_op2
        },
        {
            test_values.expected.convertion_before_op1,
            test_values.expected.convertion_before_op2_1,
            test_values.expected.convertion_before_op2_2,
            test_values.expected.convertion_after_op2,
            test_values.expected.convertion_before_result
        });
    function = function_stub.getOriginal();

    auto dummyPrecisionSelection = std::make_shared<DummyPrecisionSelection>(test_values.actual.op1, test_values.actual.op2);

    auto get_supported_precisions = [dummyPrecisionSelection](const std::shared_ptr<ngraph::Node>& op) -> ov::intel_cpu::pass::EnforcePrecision::Operation {
        return dummyPrecisionSelection->get_supported_precisions(op);;
    };

    ngraph::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::pass::EnforcePrecision>(
        element::f32,
        element::bf16,
        test_values.bf16_isa,
        get_supported_precisions);
    manager.run_passes(function);

    function_ref = function_stub.getReference();
}

namespace EnforcePrecisionTestInstantiation {
// clang-format off

std::vector<std::pair<PartialShape, PartialShape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<EnforcePrecisionParamsValues> test_cases {
    // operation #1 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        true, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16, element::bf16},
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::bf16, element::bf16}
                }
            }
        },
        {
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    // operation #1 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        false, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16, element::bf16},
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::bf16, element::bf16}
                }
            }
        },
        {
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    // operation #1 & #2 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        true, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                }
            }
        },
        {
            {},
            {},
            {element::undefined, element::bf16},
            {element::f32},
            {element::bf16}
        }
    },

    // operation #1 & #2 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        false, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                }
            }
        },
        {
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    // operation #1 & #2 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        true, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::f32}
                }
            }
        },
        {
            {},
            {},
            {},
            {element::f32},
            {element::bf16}
        }
    },

    // operation #1 & #2 supports bf16
    {
        {element::bf16, element::bf16, element::f32},
        false, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                },
            },
            {
                false, // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::f32}
                }
            }
        },
        {
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        false, // bf16_isa
        {
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {
                true,  // only_after_parameter
                true,  // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                },
            },
            {
                true,  // only_after_parameter
                false, // require_bf16_isa
                {
                    {element::bf16, element::bf16}
                }
            }
        },
        {
            {element::f32, element::f32},
            {},
            {},
            {},
            {element::bf16}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_EnforcePrecisionTest,
    EnforcePrecisionTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(test_cases)),
    EnforcePrecisionTest::getTestCaseName);

// clang-format on
} // namespace EnforcePrecisionTestInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov

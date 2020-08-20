// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include <transformations/low_precision/fuse_subtract_to_fake_quantize.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/fuse_subtract_to_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FuseSubtractToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class FuseSubtractToFakeQuantizeTransformation : public LayerTransformation,
    public testing::WithParamInterface<FuseSubtractToFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseSubtractToFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.actual.fakeQuantizeOnData,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation, ngraph::opset1::Subtract>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.expected.fakeQuantizeOnData,
            testValues.expected.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationTestValues> obj) {
        const FuseSubtractToFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.params.updatePrecisions << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization;
        return result.str();
    }
};

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareOutputs) {
    auto res = compareResults(actualFunction, referenceFunction);
    ASSERT_TRUE(res);
}

const std::vector<FuseSubtractToFakeQuantizeTransformationTestValues> testValues = {
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {element::f32}, { 128.f }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { { 128.f }, element::u8 }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    FuseSubtractToFakeQuantizeTransformation,
    ::testing::ValuesIn(testValues),
    FuseSubtractToFakeQuantizeTransformation::getTestCaseName);

} // namespace

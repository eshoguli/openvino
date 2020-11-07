// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/concat_multi_channels.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize4;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize4;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.dequantizationOperations1 << "_" <<
        values.dequantizationOperations2;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    bool,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithReshapeAtTheEndTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const bool updatePrecisions = std::get<1>(GetParam());
        const ngraph::Shape shape = std::get<2>(GetParam());
        ConcatTransformationTestValues testValues = std::get<3>(GetParam());

        testValues.params.updatePrecisions = updatePrecisions;
        if (!updatePrecisions) {
            testValues.result.fakeQuantize1.outputPrecision = testValues.actual.fakeQuantize1.outputPrecision;
            testValues.result.fakeQuantize2.outputPrecision = testValues.actual.fakeQuantize2.outputPrecision;
        }

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalWithReshapeAtTheEndTransformation(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3,
            testValues.actual.fakeQuantize4);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.actual").run_on_function(actualFunction);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::ReshapeTransformation, ngraph::opset1::Reshape>(testValues.params);
        transform.transform(actualFunction);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceWithReshapeAtTheEndTransformation(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.fakeQuantize4,
            testValues.result.dequantizationOperations1,
            testValues.result.dequantizationOperations2);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(actualFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const bool updatePrecision = std::get<1>(obj.param);
        const ngraph::Shape shape = std::get<2>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (updatePrecision ? "updatePrecision_" : "notUpdatePrecision_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithReshapeAtTheEndTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
};

const std::vector<bool> updatePrecisions = { true, false };

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ConcatWithReshapeAtTheEndTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithReshapeAtTheEndTransformation::getTestCaseName);
}  // namespace

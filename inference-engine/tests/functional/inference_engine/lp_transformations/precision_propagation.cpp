// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include <low_precision/align_concat_quantization_parameters.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precisions.hpp>
#include <low_precision/propagate_precisions.hpp>

#include <low_precision/concat.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/precision_propagation_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
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
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
    ngraph::element::Type precisionBeforeOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
    ngraph::element::Type precisionAfterOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter1;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.fakeQuantize3 << "_" <<
        values.dequantizationAfter1 << "_" <<
        values.dequantizationAfter2;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class PrecisionPropagation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::PrecisionPropagationFunction::getOriginalWithNeighbors(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.actual").run_on_function(actualFunction);

        //SimpleLowPrecisionTransformer transform;
        //if (testValues.multiChannels) {
        //    transform.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        //} else {
        //    transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        //}
        //transform.transform(actualFunction);

        auto supportedPrecisionsOnActivation = std::vector<OperationPrecisionRestriction>({
            OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

        ngraph::pass::Manager manager1;
        manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisionsOnActivation);
        manager1.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming1").run_on_function(actualFunction);

        ngraph::pass::Manager manager2;
        manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
        manager2.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming2").run_on_function(actualFunction);

        ngraph::pass::Manager manager3;
        manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        manager3.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming3").run_on_function(actualFunction);

        ngraph::pass::Manager manager4;
        manager4.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
        manager4.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transforming4").run_on_function(actualFunction);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        manager.register_pass<ngraph::pass::low_precision::ConcatTransformation>();
        manager.run_passes(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.transformed").run_on_function(actualFunction);

        referenceFunction = ngraph::builder::subgraph::PrecisionPropagationFunction::getReferenceWithNeighbors(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.precisionBeforeOp,
            testValues.result.dequantizationBefore,
            testValues.result.precisionAfterOp,
            testValues.result.dequantizationAfter1,
            testValues.result.dequantizationAfter2);

        ngraph::pass::VisualizeTree("c:\\Projects\\temp\\test.reference").run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(PrecisionPropagation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // I8: concat
    {
        LayerTransformation::createParamsI8I8(),
        false,
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-128.f}, {127.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-64}, {64.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-43}, {42.f} },
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.01f } },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    //{ 4, 3, 9, 9 }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    PrecisionPropagation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    PrecisionPropagation::getTestCaseName);
}  // namespace

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize_precision_propagation.hpp"

#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::op::FakeQuantize> makeFakeQuantize(
    const Output<Node>& parent,
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    auto generate = [](const ov::element::Type precision,
        const ngraph::Shape& shape,
        const float initialValue,
        const std::string& name) {
            const auto size = ngraph::shape_size(shape);
            std::vector<float> values(size);
            for (auto i = 0; i < size; ++i) {
                values[i] = static_cast<float>(initialValue + i);
            }
            auto constant = std::make_shared<ngraph::opset1::Constant>(precision, shape, values);
            constant->set_friendly_name(name);
            return constant;
    };

    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
        parent,
        generate(inputType, fakeQuantizeShapes[0], zeroPoint, "inputLow"),
        generate(inputType, fakeQuantizeShapes[1], 20.f, "inputHigh"),
        generate(inputType, fakeQuantizeShapes[2], zeroPoint, "outputLow"),
        generate(inputType, fakeQuantizeShapes[3], 20.f, "outputHigh"),
        256ul);
    fakeQuantize->set_friendly_name("fakeQuantize");

    return fakeQuantize;
}
}

std::shared_ptr<ov::Model> FakeQuantizePrecisionPropagationFunction::get(
    const ngraph::Shape& inputShape,
    const element::Type inputType) {
    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parameter,
        Strides{ 1, 1 }, // strides
        Shape{ 0, 0 },   // pads_begin
        Shape{ 0, 0 },   // pads_end
        Shape{ 1, 1 });  // kernel
    maxPool->set_friendly_name("maxPool");

    const auto fakeQuantize = makeFakeQuantize(
        maxPool,
        inputShape,
        inputType,
        { {}, {}, {}, {} }, // fakeQuantizeShapes,
        0.f);

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto result = std::make_shared<ngraph::opset1::Result>(fakeQuantize);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ParameterVector{ parameter }, "FakeQuantizeFunction");
    function->validate_nodes_and_infer_types();

    return function;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

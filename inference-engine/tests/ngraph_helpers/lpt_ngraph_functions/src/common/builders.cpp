// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/common/builders.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

    using namespace ngraph::pass::low_precision;

std::shared_ptr<Node> makeDequantization(
    const Output<Node>& data,
    const DequantizationOperations& dequantizationOperations) {
    Output<Node> parent = data;

    if (!dequantizationOperations.convert.empty()) {
        std::shared_ptr<ngraph::opset1::Convert> convert = dequantizationOperations.convert.addDequantizationAttribute ?
            std::make_shared<ngraph::pass::low_precision::DequantizationConvert>(data, dequantizationOperations.convert.outPrecision) :
            std::make_shared<ngraph::opset1::Convert>(data, dequantizationOperations.convert.outPrecision);
        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), convert }, convert);
        parent = convert;
    }

    if (!dequantizationOperations.subtract.empty()) {
        std::shared_ptr<ngraph::opset1::Subtract> subtract;

        std::vector<size_t> shape;
        auto values = dequantizationOperations.subtract.values;
        if (dequantizationOperations.subtract.constantShapeIsDefined) {
            shape = dequantizationOperations.subtract.constantShape;
            if (values.size() == 1ul) {
                values = std::vector<float>(shape_size(shape), values[0]);
            }
        } else {
            if (dequantizationOperations.subtract.values.size() == 1ul) {
                shape = std::vector<size_t>({});
            } else {
                const auto rank = parent.get_partial_shape().rank();
                shape = std::vector<size_t>(rank.is_dynamic() ? 4ul : rank.get_length(), 1ul);
                shape[shape.size() >= 2 ? 1ul : 0] = dequantizationOperations.subtract.values.size();
            }
        }

        std::shared_ptr<Node> subtractConst = std::make_shared<ngraph::opset1::Constant>(
            dequantizationOperations.subtract.constantPrecision != element::undefined ?
                dequantizationOperations.subtract.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        if (dequantizationOperations.subtract.addConvert) {
            std::shared_ptr<Node> subtractConstConvert = std::make_shared<ngraph::opset1::Convert>(
                subtractConst,
                dequantizationOperations.subtract.outPrecision);

            auto& rt = subtractConstConvert->get_rt_info();
            for (const std::string& attribute : dequantizationOperations.subtract.convertAttributes) {
                rt[attribute] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
            }

            subtractConst = subtractConstConvert;
        }

        Output<Node> leftBranchParent = dequantizationOperations.subtract.constantIndex == 1 ? parent : subtractConst;
        Output<Node> rightBranchParent = dequantizationOperations.subtract.constantIndex == 1 ? subtractConst : parent;

        if (((dequantizationOperations.subtract.outPrecision == element::undefined) ||
            (dequantizationOperations.subtract.outPrecision == parent.get_element_type())) &&
            (((dequantizationOperations.subtract.constantPrecision == element::undefined) ||
            (dequantizationOperations.subtract.constantPrecision == parent.get_element_type())) ||
            dequantizationOperations.subtract.addConvert)) {
            if (dequantizationOperations.subtract.constantIndex == 1ul) {
                subtract = dequantizationOperations.subtract.addDequantizationAttribute ?
                    std::make_shared<ngraph::pass::low_precision::DequantizationSubtract>(parent, subtractConst) :
                    std::make_shared<ngraph::opset1::Subtract>(parent, subtractConst);
            } else {
                subtract = dequantizationOperations.subtract.addDequantizationAttribute ?
                    std::make_shared<ngraph::pass::low_precision::DequantizationSubtract>(subtractConst, parent) :
                    std::make_shared<ngraph::opset1::Subtract>(subtractConst, parent);
            }
        } else {
            // TODO: use templates
            if (dequantizationOperations.subtract.addDequantizationAttribute) {
                if (dequantizationOperations.subtract.constantIndex == 1ul) {
                    subtract = std::make_shared<op::TypeRelaxed<ngraph::pass::low_precision::DequantizationSubtract>>(
                        std::vector<element::Type>{element::f32, element::f32},
                        std::vector<element::Type>{ element::f32 },
                        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                        ngraph::op::TemporaryReplaceOutputType(subtractConst, element::f32).get());
                } else {
                    subtract = std::make_shared<op::TypeRelaxed<ngraph::pass::low_precision::DequantizationSubtract>>(
                        std::vector<element::Type>{element::f32, element::f32},
                        std::vector<element::Type>{ element::f32 },
                        ngraph::op::TemporaryReplaceOutputType(subtractConst, element::f32).get(),
                        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get());
                }
            } else {
                if (dequantizationOperations.subtract.constantIndex == 1ul) {
                    subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(
                        std::vector<element::Type>{element::f32, element::f32},
                        std::vector<element::Type>{ element::f32 },
                        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                        ngraph::op::TemporaryReplaceOutputType(subtractConst, element::f32).get());
                } else {
                    subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(
                        std::vector<element::Type>{element::f32, element::f32},
                        std::vector<element::Type>{ element::f32 },
                        ngraph::op::TemporaryReplaceOutputType(subtractConst, element::f32).get(),
                        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get());
                }
            }

            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(subtract, dequantizationOperations.subtract.outPrecision);
        }
        if (!dequantizationOperations.subtract.addDequantizationAttribute) {
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(subtract);
        }
        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), subtract }, subtract);

        if (!dequantizationOperations.subtract.attributes.empty()) {
            auto& rt = subtract->get_rt_info();
            for (const std::string& attribute : dequantizationOperations.subtract.attributes) {
                rt[attribute] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
            }
        }

        parent = subtract;
    }

    if (!dequantizationOperations.multiply.empty()) {
        auto const newMultiply = makeMultiply(parent, dequantizationOperations.multiply);
        NetworkHelper::copyInfo({ data.get_node_shared_ptr(), newMultiply }, newMultiply);
        parent = newMultiply;
    }

    return parent.get_node_shared_ptr();
}

std::shared_ptr<Node> makeMultiply(const Output<Node>& parent, const DequantizationOperations::Multiply& multiply) {
    std::vector<size_t> shape;
    auto values = multiply.values;
    if (multiply.constantShapeIsDefined) {
        shape = multiply.constantShape;
        if (values.size() == 1ul) {
            values = std::vector<float>(shape_size(shape), values[0]);
        }
    } else {
        if (values.size() == 1ul) {
            shape = std::vector<size_t>({});
        } else {
            const auto rank = parent.get_partial_shape().rank();
            shape = std::vector<size_t>(rank.is_dynamic() ? 4ul : rank.get_length(), 1ul);
            shape[shape.size() >= 2 ? 1ul : 0] = values.size();
        }
    }

    std::shared_ptr<ngraph::opset1::Multiply> newMultiply;
    if (((multiply.outPrecision == element::undefined) ||
        (multiply.outPrecision == parent.get_element_type())) &&
        ((multiply.constantPrecision == element::undefined) ||
        (multiply.constantPrecision == parent.get_element_type()))) {
        const std::shared_ptr<ngraph::opset1::Constant> constant = std::make_shared<ngraph::opset1::Constant>(
            multiply.constantPrecision != element::undefined ?
                multiply.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        if (multiply.addDequantizationAttribute) {
            newMultiply = multiply.constantIndex == 1ul ?
                std::make_shared<ngraph::pass::low_precision::DequantizationMultiply>(parent, constant) :
                std::make_shared<ngraph::pass::low_precision::DequantizationMultiply>(constant, parent);
        } else {
            newMultiply = multiply.constantIndex == 1ul ?
                std::make_shared<ngraph::opset1::Multiply>(parent, constant) :
                std::make_shared<ngraph::opset1::Multiply>(constant, parent);
        }
    } else {
        const std::shared_ptr<ngraph::opset1::Constant> constant = std::make_shared<ngraph::opset1::Constant>(
            multiply.constantPrecision != element::undefined ?
                multiply.constantPrecision :
                parent.get_element_type(),
            shape,
            values);

        // TODO: use templates
        if (multiply.addDequantizationAttribute) {
            newMultiply = multiply.constantIndex == 1ul ?
                std::make_shared<op::TypeRelaxed<ngraph::pass::low_precision::DequantizationMultiply>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ multiply.outPrecision },
                    ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get()) :
                std::make_shared<op::TypeRelaxed<ngraph::pass::low_precision::DequantizationMultiply>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ multiply.outPrecision },
                    ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get());
        } else {
            newMultiply = multiply.constantIndex == 1ul ?
                std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ multiply.outPrecision },
                    ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get()) :
                std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(
                    std::vector<element::Type>{element::f32, element::f32},
                    std::vector<element::Type>{ multiply.outPrecision },
                    ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get());
        }
    }

    return newMultiply;
}

std::shared_ptr<Node> makeReshape(const Output<Node>& data, const Reshape& reshape) {
    auto constant = makeConstant(ngraph::element::i64, Shape({ reshape.values.size() }), reshape.values);
    return std::make_shared<ngraph::opset1::Reshape>(data, constant->output(0), reshape.special_zero);
}

std::shared_ptr<Node> makeTranspose(const Output<Node>& data, const Transpose& transpose) {
    auto constant = makeConstant(ngraph::element::i64, Shape({ transpose.values.size() }), transpose.values);
    return std::make_shared<ngraph::opset1::Transpose>(data, constant->output(0));
}

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& output,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        output,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues));
}

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const Output<ngraph::Node>& output,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<ngraph::opset1::FakeQuantize> fq = makeFakeQuantize(output, precision, fqOnData);
    return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(
        *fq,
        fqOnData.outputPrecision == element::undefined ? precision : fqOnData.outputPrecision);
}

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& input,
    const ngraph::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const bool subgraphOnConstantPath) {
    std::shared_ptr<Node> inputLowNode;
    std::shared_ptr<Node> inputHighNode;

    if (subgraphOnConstantPath) {
        const auto topConstant = ngraph::builder::makeConstant(constantPrecision, ngraph::Shape{1}, std::vector<float>(1, 0.f), false);
        const auto convert = std::make_shared<opset1::Convert>(topConstant, element::f32);

        const auto subtractMin = std::make_shared<opset1::Subtract>(
            std::make_shared<opset1::Constant>(constantPrecision, ngraph::Shape{ 1 }, std::vector<float>{fqOnData.outputLowValues[0]}),
            convert);
        const auto subtractMax = std::make_shared<opset1::Subtract>(
            std::make_shared<opset1::Constant>(constantPrecision, ngraph::Shape{ 1 }, std::vector<float>{fqOnData.outputHighValues[0]}),
            convert);

        inputLowNode = std::make_shared<opset1::Multiply>(
            std::make_shared<opset1::Constant>(
                constantPrecision,
                ngraph::Shape{ 1 },
                std::vector<float>{fqOnData.inputLowValues[0] / fqOnData.outputLowValues[0]}),
            subtractMin);
        inputHighNode = std::make_shared<opset1::Multiply>(
            std::make_shared<opset1::Constant>(
                constantPrecision,
                ngraph::Shape{ 1 },
                std::vector<float>{fqOnData.inputHighValues[0] / fqOnData.outputHighValues[0]}),
            subtractMax);
    } else {
        inputLowNode = ngraph::builder::makeConstant(
            fqOnData.inputLow.empty() ? constantPrecision : fqOnData.inputLow.outPrecision,
            fqOnData.constantShapes.empty() ? ngraph::Shape{} : fqOnData.constantShapes[0],
            fqOnData.inputLowValues,
            fqOnData.inputLowValues.empty());

        inputHighNode = ngraph::builder::makeConstant(
            fqOnData.inputHigh.empty() ? constantPrecision : fqOnData.inputHigh.outPrecision,
            fqOnData.constantShapes.empty() ?
                ngraph::Shape{} :
                (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[1]),
            fqOnData.inputHighValues,
            fqOnData.inputHighValues.empty());
    }

    const auto outputLowNode = ngraph::builder::makeConstant(
        fqOnData.outputLow.empty() ? constantPrecision : fqOnData.outputLow.outPrecision,
        fqOnData.constantShapes.empty() ?
            ngraph::Shape{} :
            (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[2]),
        fqOnData.outputLowValues,
        fqOnData.outputLowValues.empty());

    const auto outputHighNode = ngraph::builder::makeConstant(
        fqOnData.outputHigh.empty() ? constantPrecision : fqOnData.outputHigh.outPrecision,
        fqOnData.constantShapes.empty() ?
            ngraph::Shape{} :
            (fqOnData.constantShapes.size() == 1 ? fqOnData.constantShapes[0] : fqOnData.constantShapes[3]),
        fqOnData.outputHighValues,
        fqOnData.outputHighValues.empty());

    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fqOnData.quantizationLevel);

    auto& rt = fq->get_rt_info();
    for (auto& attribute : fqOnData.attributes) {
        rt[attribute->get_type_info().name] = attribute;
    }

    return fq;
}

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData) {
    const std::shared_ptr<ngraph::opset1::FakeQuantize> fq = makeFakeQuantize(input, constantPrecision, fqOnData);
    return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(
        *fq,
        fqOnData.outputPrecision == ngraph::element::undefined ? constantPrecision : fqOnData.outputPrecision);
}

std::shared_ptr<Node> addDequantizationAttribute(const std::shared_ptr<Node>& op) {
    auto& rtInfo = op->get_rt_info();
    rtInfo["DEQUANTIZATION"] = std::make_shared<VariantWrapper<DequantizationAttr>>(DequantizationAttr());
    return op;
}

void addAttributes(std::vector<std::shared_ptr<ngraph::Node>> nodes, std::vector<std::shared_ptr<Variant>> attributes) {
    for (const auto& node : nodes) {
        for (const auto& attribute : attributes) {
            auto& rt = node->get_rt_info();
            const std::string typeInfoName = attribute->get_type_info().name;
            rt[typeInfoName] = attribute;
        }
    }
}

std::shared_ptr<Node> makeConvolution(
    const std::shared_ptr<Node>& parent,
    const element::Type precision,
    const bool weightsWithoutFQ,
    const element::Type weightsprecision) {
    const size_t outputChannels = parent->get_output_partial_shape(0)[1].get_length() * 2;
    const size_t inputChannels = parent->get_output_partial_shape(0)[1].get_length();
    const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };

    std::shared_ptr<Node> weights;
    if (weightsWithoutFQ) {
        weights = std::make_shared<opset1::Constant>(weightsprecision, shape, std::vector<int>(ngraph::shape_size(shape), 100));
    } else {
        weights = ngraph::builder::makeFakeQuantize(
            std::make_shared<opset1::Constant>(precision, shape, std::vector<float>(ngraph::shape_size(shape), 1.f)),
            precision,
            255,
            { outputChannels, 1, 1, 1 },
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f),
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f));
        weights->set_friendly_name("fakeQuantizeOnWeights");
    }

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
        ngraph::op::TemporaryReplaceOutputType(weights, precision).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    convolution->set_friendly_name("convolution");

    return convolution;
}

} // namespace subgraph
} // namespace builder
} // namespace ngraph

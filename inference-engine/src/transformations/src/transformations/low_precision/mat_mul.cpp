﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mat_mul.hpp"

#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::MatMul> matMul = as_type_ptr<ngraph::opset1::MatMul>(m.get_match_root());
    if ((matMul == nullptr) || !canBeTransformed(context, matMul)) {
        return false;
    }

    //std::cout << "MatMulTransformation::transform: " << matMul->get_friendly_name() << std::endl;

    //if ((matMul->get_friendly_name() == "1321") || (matMul->get_friendly_name() == "1322")) {
    //    ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.matmul_1322").run_on_function(context.function);
    //}

    FakeQuantizeDequantization dequantization1 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 0);
    //if (!supportAsymmetricQuantization && isAsymmetricQuantization(matMul, dequantization1)) {
    //    return false;
    //}

    FakeQuantizeDequantization dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
    //if (!supportAsymmetricQuantization && isAsymmetricQuantization(matMul, dequantization2)) {
    //    return false;
    //}

    matMul = as_type_ptr<ngraph::opset1::MatMul>(separateInStandaloneBranch(matMul));
    dequantization1 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 0);
    dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);

    if (dequantization2.empty()) {
        const std::shared_ptr<opset1::FakeQuantize> fakeQuantize =
            as_type_ptr<opset1::FakeQuantize>(dequantization2.data.get_node_shared_ptr());
        if (fakeQuantize != nullptr) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
            const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, true);

            auto tuple = NetworkHelper::decomposeFakeQuantize(
                fakeQuantize,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);

            dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
        }
    }

    const std::shared_ptr<opset1::MatMul> newMatMul = std::make_shared<ngraph::op::TypeRelaxed<opset1::MatMul>>(
        std::vector<element::Type>({ element::f32, element::f32 }), std::vector<element::Type>({}),
        ngraph::op::TemporaryReplaceOutputType(dequantization1.data, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantization2.data, element::f32).get(),
        matMul->get_transpose_a(),
        matMul->get_transpose_b());
    NetworkHelper::setOutDataPrecisionForTypeRelaxed(newMatMul, matMul->get_output_element_type(0));

    //if (matMul->get_friendly_name() == "1322") {
    //    ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.matmul_1322").run_on_function(context.function);
    //}


    auto transpose = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<Node> {
        const Shape outputShape = node->get_output_shape(0);

        std::vector<size_t> transposeConstant(outputShape.size());
        std::iota(transposeConstant.begin(), transposeConstant.end(), 0);
        std::swap(*(transposeConstant.end() - 1), *(transposeConstant.end() - 2));

        std::shared_ptr<Node> transposedConstant = fold<ngraph::opset1::Transpose>(
            node,
            opset1::Constant::create(element::i64, Shape{ transposeConstant.size() }, transposeConstant));
        return transposedConstant;
    };

    const std::shared_ptr<Node> const1 = matMul->get_transpose_a() ?
        transpose(dequantization1.multiply->get_input_node_shared_ptr(1)) :
        dequantization1.multiply->get_input_node_shared_ptr(1);

    std::shared_ptr<Node> const2 = matMul->get_transpose_b() ?
        transpose(dequantization2.multiply->get_input_node_shared_ptr(1)) :
        dequantization2.multiply->get_input_node_shared_ptr(1);

    const std::shared_ptr<opset1::Multiply> newMultiply = std::make_shared<opset1::Multiply>(
        newMatMul,
        NetworkHelper::toScalarIfPossible(fold<ngraph::opset1::Multiply>(const1, const2)));

    //Shape const2Shape = const2->output(0).get_shape();
    //if ((newMatMul->output(0).get_shape().size() - const2Shape.size()) == 1ul) {
    //    const2Shape.insert(const2Shape.begin(), 1ul);
    //    const2 = std::make_shared<opset1::Constant>(const2->output(0).get_element_type(), const2Shape, as_type_ptr<opset1::Constant>(const2)->get_data_ptr());
    //}

    //if (matMul->get_friendly_name() == "1345") {
    //    ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.tmp").run_on_function(context.function);
    //}

    //const std::shared_ptr<Node> multiplyConstant = NetworkHelper::toScalarIfPossible(fold<ngraph::opset1::Multiply>(const1, const2));
    //const auto shape1 = newMatMul->output(0).get_shape();
    //const auto shape2 = multiplyConstant->output(0).get_shape();
    //const std::shared_ptr<opset1::Multiply> newMultiply = std::make_shared<opset1::Multiply>(newMatMul, multiplyConstant);
    replace_node(matMul, newMultiply);

    updateOutput(context, newMultiply, matMul);

    //if (matMul->get_friendly_name() == "1322") {
    //    ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.matmul_1322").run_on_function(context.function);
    //}

    return true;
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::Multiply>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::FakeQuantize>() }));
}

bool MatMulTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    return true;
}

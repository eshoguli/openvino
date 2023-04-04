// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_convert_transpose_convert.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/convert_saturation.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ngraph::Function> SubgraphConvertTransposeConvertFunction::get(
    const ngraph::PartialShape& input_shape,
    const ngraph::element::Type input_precision,
    const std::vector<size_t>& transpose_order,
    const element::Type convert_before_precision,
    const element::Type convert_after_precision1,
    const element::Type convert_after_precision2) {
    const auto create_convert = [](std::shared_ptr<Node> parent, const element::Type convertion_type) -> std::shared_ptr<Node> {
        return convertion_type == element::undefined
            ? std::dynamic_pointer_cast<Node>(parent)
            : std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convertion_type);
    };

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(input_precision, input_shape);
    parameter->set_friendly_name("parameter");

    std::shared_ptr<Node> parent = create_convert(parameter, convert_before_precision);

    parent = std::make_shared<ngraph::opset1::Transpose>(
        parent,
        std::make_shared<op::v0::Constant>(ov::element::i32, Shape{ transpose_order.size() }, transpose_order));

    parent = create_convert(parent, convert_after_precision1);
    parent = create_convert(parent, convert_after_precision2);

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    auto& result_out_tensor = result->get_output_tensor(0);
    result_out_tensor.set_names({ "result_tensor" });
    result->set_friendly_name("result");

    const ngraph::ResultVector results{ result };
    const ngraph::ParameterVector parameters{ parameter };
    const auto model = std::make_shared<ngraph::Function>(results, parameters, "SubgraphConvertTransposeConvertFunction");
    return model;
}

std::shared_ptr<ov::Model> SubgraphConvertTransposeConvertFunction::initOriginal() const {
    return get(
        input_shapes[0],
        input_precision,
        transpose_order,
        original.convert_before_precision,
        original.convert_after_precision);
}

std::shared_ptr<ov::Model> SubgraphConvertTransposeConvertFunction::initReference() const {
    return get(
        input_shapes[0],
        input_precision,
        transpose_order,
        expected.convert_before_precision,
        expected.convert_after_precision1,
        expected.convert_after_precision2);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

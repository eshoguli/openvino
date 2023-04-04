// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "ngraph/opsets/opset1.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class SubgraphConvertTransposeConvertFunction : public SnippetsFunctionBase {
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

    explicit SubgraphConvertTransposeConvertFunction(
        const std::vector<PartialShape> input_shapes,
        const ngraph::element::Type input_precision,
        const std::vector<size_t>& transpose_order,
        const Original original,
        const Expected expected) :
        SnippetsFunctionBase(input_shapes),
        input_precision(input_precision),
        transpose_order(transpose_order),
        original(original),
        expected(expected) {
        OPENVINO_ASSERT(input_shapes.size() == 1ull, "input_shapes size has to be equal to 2");
    }

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::PartialShape& input_shape,
        const ngraph::element::Type input_precision,
        const std::vector<size_t>& transpose_order,
        const element::Type convert_before_precision,
        const element::Type convert_after_precision1,
        const element::Type convert_after_precision2 = element::undefined);

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    const ngraph::element::Type input_precision;
    const std::vector<size_t> transpose_order;
    const Original original;
    const Expected expected;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

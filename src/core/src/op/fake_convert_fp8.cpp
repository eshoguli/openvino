// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/op/fake_convert_fp8.hpp"

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/fake_quantize.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

op::FakeConvertFp8::FakeConvertFp8() : BaseFakeQuantize(), m_levels() {}

op::FakeConvertFp8::FakeConvertFp8(const Output<Node>& data, const Output<Node>& scale)
    : BaseFakeQuantize({data, scale}) {
    constructor_validate_and_infer_types();
}

void op::FakeConvertFp8::validate_and_infer_types() {
    OV_OP_SCOPE(v0_FakeConvertFp8_validate_and_infer_types);
    ov::PartialShape data_pshape = get_input_partial_shape(0);

    for (auto i = 1; i <= 4; i++) {
        if (m_auto_broadcast.m_type == op::AutoBroadcastType::NONE) {
            NODE_VALIDATION_CHECK(this,
                                  ov::PartialShape::merge_into(data_pshape, get_input_partial_shape(i)),
                                  "Argument shapes are inconsistent.");
        } else if (m_auto_broadcast.m_type == op::AutoBroadcastType::NUMPY ||
                   m_auto_broadcast.m_type == op::AutoBroadcastType::PDPD) {
            NODE_VALIDATION_CHECK(
                this,
                ov::PartialShape::broadcast_merge_into(data_pshape, get_input_partial_shape(i), m_auto_broadcast),
                "Argument shapes are inconsistent.");
        } else {
            NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool ngraph::op::v0::FakeConvertFp8::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_FakeConvertFp8_visit_attributes);
    visitor.on_attribute("levels", m_levels);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

shared_ptr<Node> op::FakeConvertFp8::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_FakeConvertFp8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<FakeConvertFp8>(new_args.at(0), new_args.at(1));
}

bool ngraph::op::FakeConvertFp8::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_FakeConvertFp8_evaluate);
    return true;
}

bool ngraph::op::FakeConvertFp8::has_evaluate() const {
    OV_OP_SCOPE(v0_FakeConvertFp8_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

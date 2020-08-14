//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::BinaryElementwiseArithmetic, "BinaryElementwiseArithmetic", 0);

#ifdef LPT_SUPPORT
bool op::util::BinaryElementwiseArithmetic::multi_type_global = false;
#endif

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob)
    : m_autob(autob)
{
}

#ifdef LPT_SUPPORT
op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob,
                                                                   const bool multi_type)
    : Op({arg0, arg1})
    , m_autob(autob)
    , m_multi_type(multi_type)
{
}
#else
op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}
#endif

void op::util::BinaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic(
    const op::AutoBroadcastSpec& autob)
{
#ifdef LPT_SUPPORT
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, autob, m_multi_type);
#else
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, autob);
#endif

    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          args_et,
                          ").");

#ifdef LPT_SUPPORT
    if (m_multi_type)
    {
        element::Type element_type1 = get_input_element_type(0);
        element::Type element_type2 = get_input_element_type(1);
        args_et =
            element_type1.bitwidth() > element_type2.bitwidth() ? element_type1 : element_type2;
    }
#endif
    set_output_type(0, args_et, args_pshape);
}

void op::util::BinaryElementwiseArithmetic::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic(m_autob);
}

bool op::util::BinaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/base_fake_quantize.hpp"

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

op::BaseFakeQuantize::BaseFakeQuantize() : Op() {}

op::BaseFakeQuantize::BaseFakeQuantize(const ov::OutputVector& args) : Op(args) {}


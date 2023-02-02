// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace v0 {
///
/// \brief      Class performing element-wise linear quantization.
///
/// \note       Input floating point values are quantized into a discrete
///             set of floating point values.
///
/// \paragraph Implementation This class creates a node which performs the following
///            operation:
///
///            round((data - input_low) / (input_high - input_low) * (levels-1)) /
///                 (levels-1) * (output_high - output_low) + output_low
///
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BaseFakeQuantize : public Op {
public:
    BaseFakeQuantize();
    BaseFakeQuantize(const ov::OutputVector& args);
};

}  // namespace v0
}  // namespace op
}  // namespace ov

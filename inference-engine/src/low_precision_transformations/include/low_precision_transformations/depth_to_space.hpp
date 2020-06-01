// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "ie_layers.h"
#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/transparent_base_transformation.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(DepthToSpaceTransformation) : public TransparentBaseTransformation {
public:
    DepthToSpaceTransformation(const Params& params) : TransparentBaseTransformation(params) {}
    ~DepthToSpaceTransformation() override {}
    void transform(TransformationContext& context, CNNLayer& layer) const override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const override;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine

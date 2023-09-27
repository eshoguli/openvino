// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ov {
/**
 * @ingroup ie_transformation_common_api
 * @brief Check if clean is disabled on @ref Node.
 *
 * For more details about the attribute, refer to
 * [DisableCleanupAttribute](@ref openvino_docs_OV_UG_lpt_DisableCleanup) page in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API DisableCleanupAttribute : public SharedAttribute<std::vector<ov::element::Type>> {
public:
    OPENVINO_RTTI("LowPrecision::DisableCleanup", "", ov::RuntimeAttribute);
    DisableCleanupAttribute() = default;

    static ov::Any create(const std::shared_ptr<ov::Node>& node) {
        auto& rt = node->get_rt_info();
        return (rt[DisableCleanupAttribute::get_type_info_static()] = DisableCleanupAttribute());
    }

    bool is_copyable() const override {
        return false;
    }
};
} // namespace ov

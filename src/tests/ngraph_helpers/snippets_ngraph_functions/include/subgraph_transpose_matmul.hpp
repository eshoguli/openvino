// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "openvino/core/model.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class SubgraphTransposeMatMulFunction : public SnippetsFunctionBase {
public:
    SubgraphTransposeMatMulFunction(
        const std::vector<ov::PartialShape>& input_shapes,
        const element::Type input_type,
        const bool transpose,
        const bool mat_mul);

protected:
    std::shared_ptr<Model> initOriginal() const override;

private:
    const bool transpose;
    const bool mat_mul;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

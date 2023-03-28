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

    /*
     * Don't call this method explicity. You should create the instance of PrecisionPropagationConvertionFunction before.
     * After the method will be called implicitly in getOriginal.
     * Note, please, getReference and getLowered methods are not implemented and throw exception.
     */
    static std::shared_ptr<ov::Model> get(
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

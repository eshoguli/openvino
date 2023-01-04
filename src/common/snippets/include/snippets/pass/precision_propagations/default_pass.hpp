// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>
#include <snippets/itt.hpp>
#include "snippets/op/convert_saturation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagations {

using default_pass_callback = std::function<bool(const std::shared_ptr<Node>& node)>;

class DefaultPass : public ngraph::pass::PassBase {
public:
    virtual ~DefaultPass() = default;

    default_pass_callback get_callback() const noexcept {
        return m_callback;
    }

protected:
    void register_callback(const default_pass_callback& callback);

private:
    default_pass_callback m_callback;
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

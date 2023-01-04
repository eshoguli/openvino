// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>
#include "snippets/pass/precision_propagations/default_pass.hpp"
#include "snippets/pass/precision_propagations/sequential_graph_rewrite.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define LP_TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define SNIPPETS_API OPENVINO_CORE_EXPORTS
#    else
#        define SNIPPETS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ngraph {
namespace snippets {
namespace pass {
namespace precision_propagations {

class SequentialManager {
public:
    SequentialManager();
    ~SequentialManager() = default;

    void run_passes(std::shared_ptr<ov::Model>);

    template <typename T, bool Enable = true, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<ov::pass::PassBase>(pass);
        m_pass_list.push_back(pass_base);
        return nullptr;
    }

    template <typename T, bool Enable = true, class... Args>
    std::shared_ptr<T> register_default_pass(Args&&... args) {
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_base = std::static_pointer_cast<DefaultPass>(pass);
        m_default_pass_list.push_back(pass_base);
        return nullptr;
    }

private:
    std::vector<std::shared_ptr<ov::pass::PassBase>> m_pass_list;
    std::vector<std::shared_ptr<DefaultPass>> m_default_pass_list;
};

}  // namespace precision_propagations
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph

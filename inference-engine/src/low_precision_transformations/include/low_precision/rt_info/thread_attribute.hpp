// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>
#include <thread>

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/layer_transformation.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

//#ifndef NDEBUG
//#define DEBUG_THREADING
//#endif

//#define THREAD_BY_BRANCH

namespace ngraph {

class LP_TRANSFORMATIONS_API CompletionCounter {
public:
    explicit CompletionCounter(const size_t count);
    bool complete();
    bool isCompleted() noexcept;
    size_t getTotalCount() noexcept;
    size_t getNotCompletedCount() noexcept;

private:
    size_t totalCount;
    // to debug
    size_t notCompletedCount;
    std::mutex mutex;
};

class LP_TRANSFORMATIONS_API ThreadAttribute {
public:
    explicit ThreadAttribute(const size_t thread_id);
    ThreadAttribute(const size_t thread_id, const size_t input_thread_id);
#ifdef DEBUG_THREADING
    std::unordered_set<size_t> input_thread_ids;
#endif
    std::unordered_set<size_t> output_thread_ids;
    size_t thread_id;
    bool handled;
#ifdef DEBUG_THREADING
    std::__thread_id handled_thread_id;
#endif
    std::shared_ptr<CompletionCounter> completion_counter;
};

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<ThreadAttribute>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<ThreadAttribute> : public VariantImpl<ThreadAttribute> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::Thread", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    ThreadAttribute& get() { return this->m_value; }

    // merge attribute instances which can be got from different sources: node, input port or output port
    void merge(std::vector<std::shared_ptr<VariantWrapper<ThreadAttribute>>>& attributes);
    // visualize shared attributes details in VizualizeTree pass
    std::string to_string() override;
};
} // namespace ngraph

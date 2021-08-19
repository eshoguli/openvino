// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/thread_attribute.hpp"

#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

ThreadAttribute::ThreadAttribute(const size_t thread_id) : thread_id(thread_id) {
}

ThreadAttribute::ThreadAttribute(const size_t thread_id, const size_t input_thread_id) :
    thread_id(thread_id),
    input_thread_ids({input_thread_id}),
    handled(false),
    handled_thread_id() {
}

template class ngraph::VariantImpl<ThreadAttribute>;

constexpr VariantTypeInfo VariantWrapper<ThreadAttribute>::type_info;

void VariantWrapper<ThreadAttribute>::merge(std::vector<std::shared_ptr<VariantWrapper<ThreadAttribute>>>& attributes) {
}

namespace thread_attribute {
std::string to_string(const std::unordered_set<size_t>& thread_ids) {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto id : thread_ids) {
        if (!first) {
            ss << ", ";
        }
        ss << id;
        first = false;
    }
    ss << "}";
    return ss.str();
}
} // namespace thread_attribute

std::string VariantWrapper<ThreadAttribute>::to_string() {
    std::stringstream ss;
    ss << "thread_id: " << m_value.thread_id <<
        ", in: " << thread_attribute::to_string(m_value.input_thread_ids) <<
        ", out: " << thread_attribute::to_string(m_value.output_thread_ids) <<
        ", handled: " << m_value.handled <<
        ", handled_thread_id: " << m_value.handled_thread_id;
    return ss.str();
}

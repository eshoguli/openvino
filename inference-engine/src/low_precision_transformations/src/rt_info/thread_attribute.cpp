// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/thread_attribute.hpp"

#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

ThreadAttribute::ThreadAttribute(const size_t thread_id) : input_thread_ids({thread_id}), thread_id(thread_id) {
}

template class ngraph::VariantImpl<ThreadAttribute>;

constexpr VariantTypeInfo VariantWrapper<ThreadAttribute>::type_info;

void VariantWrapper<ThreadAttribute>::merge(std::vector<std::shared_ptr<VariantWrapper<ThreadAttribute>>>& attributes) {
}

std::string VariantWrapper<ThreadAttribute>::to_string() {
    std::stringstream ss;
    ss << "thread_id: " << m_value.thread_id << ", input_thread_ids: {";
    bool first = true;
    for (const auto id : m_value.input_thread_ids) {
        if (!first) {
            ss << ", ";
        }
        ss << id;
        first = false;
    }
    ss << "}";
    return ss.str();
}

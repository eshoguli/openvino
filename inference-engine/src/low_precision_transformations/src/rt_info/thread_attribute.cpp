// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/thread_attribute.hpp"

#include <vector>
#include <mutex>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

CompletionCounter::CompletionCounter(const size_t count) : totalCount(count), notCompletedCount(count) {
}

bool CompletionCounter::complete() {
    const std::lock_guard<std::mutex> lock(mutex);
    assert(notCompletedCount != 0);
    notCompletedCount--;
    return notCompletedCount == 0;
}

bool CompletionCounter::isCompleted() noexcept {
    const std::lock_guard<std::mutex> lock(mutex);
    return notCompletedCount == 0;
}

size_t CompletionCounter::getTotalCount() noexcept {
    const std::lock_guard<std::mutex> lock(mutex);
    return totalCount;
}
size_t CompletionCounter::getNotCompletedCount() noexcept {
    const std::lock_guard<std::mutex> lock(mutex);
    return notCompletedCount;
}

ThreadAttribute::ThreadAttribute(const size_t thread_id) : thread_id(thread_id) {
}

ThreadAttribute::ThreadAttribute(const size_t thread_id, const size_t input_thread_id) :
    thread_id(thread_id),
#ifdef c
    input_thread_ids({input_thread_id}),
#endif
    handled(false) {
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
    std::stringstream ss2;
    if (m_value.completion_counter == nullptr) {
        ss2 << "{}";
    } else {
        ss2 << "{completed=" << (m_value.completion_counter->isCompleted() ? "true" : "false") <<
            ", count=" << m_value.completion_counter->getNotCompletedCount() << "/" << m_value.completion_counter->getTotalCount() << "}";
    }
    std::stringstream ss;
    ss << "thread_id: " << m_value.thread_id <<
#ifdef DEBUG_THREADING
        ", in: " << thread_attribute::to_string(m_value.input_thread_ids) <<
#endif
        ", out: " << thread_attribute::to_string(m_value.output_thread_ids) <<
        ", handled: " << (m_value.handled ? "true" : "false") <<
        ", completion_counter: " << ss2.str();

#ifdef DEBUG_THREADING
    ss << ", handled_thread_id: " << m_value.handled_thread_id;
#endif
    return ss.str();
}

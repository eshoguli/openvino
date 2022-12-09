// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {

//class jit_snippets_generator : public dnnl::impl::cpu::x64::jit_generator {
//public:
//    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)
//
//    ~jit_snippets_generator() = default;
//
//    jit_snippets_generator() : jit_generator() {
//    }
//
//    void generate() override {
//    }
//
//    // TODO: Xbyak::LabelManager - can we use it?
//    void L(std::shared_ptr<Xbyak::Label>& label, const size_t id) {
//        //auto it = labels.find(id);
//        //if (it != labels.end()) {
//        //    throw ov::Exception("label with name already exists");
//        //}
//
//        if (exists_label(id)) {
//            throw ov::Exception("label with name already exists");
//        }
//
//        // label is initialized here
//        dnnl::impl::cpu::x64::jit_generator::L(*label);
//
//        //// store initialized label
//        //auto res = labels.emplace(id, label);
//        //if (!res.second) {
//        //    throw ov::Exception("label with name already exists");
//        //}
//
//        add_label(id, label);
//    }
//
//    void add_label(const size_t id, std::shared_ptr<Xbyak::Label>& label) {
//        auto it = labels.find(id);
//        if (it != labels.end()) {
//            throw ov::Exception("label with name already exists");
//        }
//
//        // store initialized label <= ?
//        auto res = labels.emplace(id, label);
//        if (!res.second) {
//            throw ov::Exception("label with name already exists");
//        }
//    }
//
//    bool exists_label(const size_t id) {
//        auto it = labels.find(id);
//        return it != labels.end();
//    }
//
//    std::shared_ptr<Xbyak::Label> get_label(const size_t id) {
//        auto it = labels.find(id);
//        if (it == labels.end()) {
//            throw ov::Exception("label is absent");
//        }
//        return it->second;
//    }
//
//    // TODO: free_label?
//
//    void init_registers(const std::vector<size_t>& regs) {
//        if (!free_registers.empty()) {
//            throw ov::Exception("registers are not empty");
//        }
//        if (!allocated_named_registers.empty()) {
//            throw ov::Exception("allocated registers are not empty");
//        }
//        std::copy(regs.begin(), regs.end(), inserter(free_registers, free_registers.begin()));
//    }
//
//    int alloc_register(const size_t unique_key = -1) {
//        if (free_registers.size() == 0ul) {
//            throw ov::Exception("not enough registers");
//        }
//        if ((unique_key != static_cast<size_t>(-1)) && (allocated_named_registers.find(unique_key) != allocated_named_registers.end())) {
//            throw ov::Exception("register with name '" + std::to_string(unique_key) + "' has been allocated already");
//        }
//
//        auto reg_it = free_registers.begin();
//        auto reg = *reg_it;
//        free_registers.erase(reg_it);
//
//        if (unique_key != -1) {
//            allocated_named_registers.emplace(unique_key, reg);
//        }
//        return reg;
//    }
//
//    int get_register(const size_t unique_key) {
//        auto reg_it = allocated_named_registers.find(unique_key);
//        if (reg_it == allocated_named_registers.end()) {
//            throw ov::Exception("register with name '" + std::to_string(unique_key) + "' was not found");
//        }
//        return reg_it->second;
//    }
//
//    void free_register(int reg) {
//        if (free_registers.find(reg) != free_registers.end()) {
//            throw ov::Exception("register was not allocated");
//        }
//        free_registers.insert(reg);
//
//        for (auto it = allocated_named_registers.begin(); it != allocated_named_registers.end(); ++it) {
//            if (it->second == reg) {
//                allocated_named_registers.erase(it);
//                break;
//            }
//        }
//    }
//
//private:
//    //LabelManager label_manager;
//    std::unordered_map<size_t, std::shared_ptr<Xbyak::Label>> labels;
//
//    std::set<int> free_registers;
//    std::unordered_map<size_t, int> allocated_named_registers;
//};

}   // namespace intel_cpu
}   // namespace ov

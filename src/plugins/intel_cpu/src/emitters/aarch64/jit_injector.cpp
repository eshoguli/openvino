// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_injector.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

#define IDX(a) static_cast<uint32_t>(a.getIdx())

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

// template <cpu_isa_t isa>
// jit_injector_f32<isa>::jit_injector_f32(jit_generator* host,
//                                         const Xbyak_aarch64::XReg x_table) : h(host), x_table(x_table) {
// }

template <cpu_isa_t isa>
void jit_injector_f32<isa>::emit() const {
    h->align(64);

    h->L(*l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template struct jit_injector_f32<dnnl::impl::cpu::aarch64::asimd>;

} // namespace aarch64
} // namespace intel_cpu
} // namespace ov

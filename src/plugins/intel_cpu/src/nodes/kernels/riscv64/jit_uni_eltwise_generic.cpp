// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

#include "emitters/plugin/riscv64/jit_add_emitter.hpp"
#include "emitters/plugin/riscv64/jit_divide_emitter.hpp"
#include "emitters/plugin/riscv64/jit_multiply_emitter.hpp"
#include "emitters/plugin/riscv64/jit_power_static_emitter.hpp"
#include "emitters/plugin/riscv64/jit_subtract_emitter.hpp"

// TODO: debug only
#include "openvino/util/env_util.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

void jit_uni_eltwise_kernel::operator()(
    const node::jit_eltwise_call_args_ptrs* const_args,
    const jit_eltwise_call_args_indexes* indexes) {
    assert(ker_ && "jit_uni_eltwise_kernel: ker_ is null");
//    if (print_tensors) {
//        for (auto i = 0; i < MAX_ELTWISE_DIM_RANK; ++i) {
//            std::cout << indexes->indexes[i] << ", ";
//        }
//        std::cout << std::endl;
//    }

    //std::cout << "debug: jit_uni_eltwise_kernel::operator: start" << std::endl;
    ker_(const_args, indexes);
    //std::cout << "debug: jit_uni_eltwise_kernel::operator: end" << std::endl;
}

jit_uni_eltwise_generic::jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                                 const std::vector<EltwiseData>& eltwise_data,
                                                 const std::vector<ov::intel_cpu::Type>& ops_list,
                                                 const dnnl::post_ops& post_ops) :
                                                 jit_uni_eltwise_kernel(jep),
                                                 jit_generator(),
                                                 eltwise_data_(eltwise_data),
                                                 ops_list_(ops_list),
                                                 post_ops_(post_ops) {
    print_tensors = ov::util::getenv_bool("OV_PRINT_TENSORS");
    jit_compute = ov::util::getenv_bool("OV_JIT_COMPUTE", true);
    jit_loop = ov::util::getenv_bool("OV_JIT_LOOP", true);
}

namespace {
std::ostream& operator<<(std::ostream& os, const Algorithm& algorithm) {
    switch (algorithm) {
        case Algorithm::EltwiseMulAdd:
            os << "EltwiseMulAdd";
            break;
        case Algorithm::EltwiseAdd:
            os << "EltwiseAdd";
            break;
        case Algorithm::EltwiseDivide:
            os << "EltwiseDivide";
            break;
        case Algorithm::EltwiseMultiply:
            os << "EltwiseMultiply";
            break;
        case Algorithm::EltwisePowerStatic:
            os << "EltwisePowerStatic";
            break;
        case Algorithm::EltwiseSigmoid:
            os << "EltwiseSigmoid";
            break;
        case Algorithm::EltwiseExp:
            os << "EltwiseExp";
            break;
        case Algorithm::EltwiseGeluErf:
            os << "EltwiseGeluErf";
            break;
        case Algorithm::EltwiseGeluTanh:
            os << "EltwiseGeluTanh";
            break;
        case Algorithm::EltwiseSwish:
            os << "EltwiseSwish";
            break;
        case Algorithm::EltwiseHswish:
            os << "EltwiseHswish";
            break;
        case Algorithm::EltwiseEqual:
            os << "EltwiseEqual";
            break;
        case Algorithm::EltwiseTanh:
            os << "EltwiseTanh";
            break;
        default:
            os << "[other]";
            break;
    }
    return os;
}

    std::ostream &operator<<(std::ostream &os, const LMUL lmul) {
    switch (lmul) {
        case LMUL::m1:
            os << "LMUL::m1";
            break;
        case LMUL::m2:
            os << "LMUL::m2";
            break;
        case LMUL::m4:
            os << "LMUL::m4";
            break;
        case LMUL::m8:
            os << "LMUL::m8";
            break;
        default:
            OPENVINO_ASSERT(false, "unknown LMUL");
    }
    return os;
}
} // namespace

void jit_uni_eltwise_generic::generate() {
//    if (print_tensors) {
//        std::cout << "jit_uni_eltwise_generic: generate" << std::endl;
//    }

    preamble();

    //auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);
    auto const exec_prc = ov::element::f32;

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    size_t input_count = eltwise_emitter->get_inputs_count();
    size_t max_aux_count = eltwise_emitter->get_aux_vecs_count();
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        const auto emitter = create_eltwise_emitter(eltwise_data_[i], exec_prc);
        post_op_emitters.push_back(emitter);

        // first input is destination register
        input_count += emitter->get_inputs_count() - 1;
        max_aux_count = std::max(max_aux_count, emitter->get_aux_vecs_count());
    }

    if (print_tensors) {
        if (eltwise_data_.size() > 1ul) {
            std::cout << "Fused with: ";
            for (const auto &data : eltwise_data_) {
                std::cout << data.algo << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Not fused" << std::endl;
        }
    }

    const auto getLmul = [](const size_t reg_count) {
        const auto multiplier = static_cast<size_t>(32 / reg_count);
        if (multiplier <= 1) {
            return LMUL::m1;
        } else if ((multiplier == 2) || (multiplier == 3)) {
            return LMUL::m2;
        } else if ((4 <= multiplier) && (multiplier <= 7)) {
            return LMUL::m4;
        } else {
            return LMUL::m8;
        }
    };

    // TODO: destination register can be used as input register: refactor later
    const size_t vec_registers_count = input_count + max_aux_count + 1; // 1 vector register for output
    const size_t vmm_dst_idx = vec_registers_count - 1;
    const LMUL lmul = getLmul(vec_registers_count);
    std::cout << "vec_registers_count=" << vec_registers_count << ", lmul=" << lmul << std::endl;

    const auto &jep = jep_;

    Reg param1 = Xbyak_riscv::a0;
    Reg param2 = Xbyak_riscv::a1;

    Reg reg_post_op_ptrs = t0;
    Reg start_to_offsets = reg_post_op_ptrs;

    //Reg reg_oc_off = t1;
    Reg reg_const_params = param1;
    Reg reg_indexes = param2;

    const int offset_count = jep.input_size - 1;

    // ptrs initializing
    if (jep.use_runtime_ptrs) {
//        if (print_tensors) {
//            std::cout << "jit_uni_eltwise_generic: runtime ptr" << std::endl;
//        }

        for (size_t i = 0; i < jep.inputs_number; i++) {
            ld(start_to_offsets, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_offsets) + i * sizeof(size_t)));
            ld(get_src_reg(i), reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[0]) + i * sizeof(size_t)));

            Reg offset_reg = get_aux_gpr(0);
            Reg index_reg = get_aux_gpr(1);
            for (int j = 0; j < offset_count; j++) {
                ld(offset_reg, start_to_offsets, static_cast<int32_t>(j * sizeof(size_t)));
                ld(index_reg, reg_indexes, static_cast<int32_t>(j * sizeof(size_t)));
                mul(offset_reg, offset_reg, index_reg);
                add(get_src_reg(i), offset_reg, get_src_reg(i));
            }
        }

        ld(start_to_offsets, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_offsets)));
        ld(reg_dst, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr)));
        Reg offset_reg = get_aux_gpr(0);
        Reg index_reg = get_aux_gpr(1);
        for (int j = 0; j < offset_count; j++) {
            ld(offset_reg, start_to_offsets, static_cast<int32_t>(j * sizeof(size_t)));
            ld(index_reg, reg_indexes, static_cast<int32_t>(j * sizeof(size_t)));
            mul(offset_reg, offset_reg, index_reg);
            add(reg_dst, offset_reg, reg_dst);
        }

        // mov(reg_oc_off, 0);

        ld(reg_work_amount, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, work_amount)));
    } else {
//        if (print_tensors) {
//            std::cout << "jit_uni_eltwise_generic: static ptr" << std::endl;
//        }

        auto init_ptrs_with_offsets = [this, offset_count, param2](Reg pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    Reg offset_reg(get_aux_gpr(0));
//                    if (print_tensors) {
//                        std::cout << "jit_uni_eltwise_generic: offsets[" << j << "] = " << offsets[j] << std::endl;
//                    }
                    // TODO: static cast size_t ot int: add assert
                    li(offset_reg, static_cast<int>(offsets[j]));
                    Reg index_reg(get_aux_gpr(1));
                    ld(index_reg, param2, static_cast<int32_t>(j * sizeof(size_t)));

                    mul(offset_reg, offset_reg, index_reg);
                    add(pointer, pointer, offset_reg);

//                    if (print_tensors) {
//                        std::cout << "init_ptrs_with_offsets: offset=" << (offsets[j] * j * sizeof(size_t))
//                                  << std::endl;
//                    }
                }
            }
        };

        for (size_t i = 0; i < jep.inputs_number; i++) {
            const auto src_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t));
//            if (print_tensors) {
//                std::cout << "debug: src_offset" << i << " = " << src_offset << std::endl;
//            }
            ld(get_src_reg(i), param1, src_offset);
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
        }

        const auto dst_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr));
//        if (print_tensors) {
//            std::cout << "debug: dst_offset = " << dst_offset << std::endl;
//        }

        // TODO: refactor: a0 - hardcode
        ld(reg_dst, a0, dst_offset);
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        addi(reg_work_amount, x0, jep.work_amount); // li
    }

    Label unroll_loop_label;
    Label unroll_loop_label2;
    Label unroll_loop_end_label;
    Label main_loop_label;
    Label main_loop_end_label;
    Label tail_loop_label;
    Label tail_loop_end_label;

    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            if (print_tensors) {
                std::cout << "input " << i << ": scalar" << std::endl;
            }
            // TODO: move outside of loop
            vsetvli(t0, reg_work_amount, SEW::e32, lmul);

            // TODO: workaround: broadcast load
            flw(f0, get_src_reg(i));
            const auto vmm_reg = get_vmm_reg(i, lmul);
            vxor_vv(vmm_reg, vmm_reg, vmm_reg);
            vfadd_vf(vmm_reg, vmm_reg, f0);
        }
    }

    size_t min_src_size = jep.dst_size;
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] != 1)
            min_src_size = std::min(min_src_size, jep.src_size[i]);
    }
    if (jep_.oc_size > 1)
        min_src_size = std::min(min_src_size, jep_.oc_size);

//    if (print_tensors) {
//        std::cout << "jep.work_amount=" << jep.work_amount << std::endl;
//        std::cout << "jep.dst_size: " << jep.dst_size << std::endl;
//        std::cout << "min_src_size: " << min_src_size << std::endl;
//    }

    if (min_src_size != jep.dst_size) {
//        if (print_tensors) {
//            std::cout << "jit_uni_eltwise_generic: unroll" << std::endl;
//        }

        bool is_valid_configuration = true;
        if (jep.dst_size % min_src_size != 0)
            is_valid_configuration = false;

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                is_valid_configuration = false;
        }

        if (jep.oc_size > 1 && jep.oc_size != min_src_size && jep.oc_size != jep.dst_size)
            is_valid_configuration = false;

        if (!is_valid_configuration)
            OPENVINO_THROW("Eltwise jitter has invalid configuration for Eltwise node");

        //addi(t1, x0, min_src_size);
        // TODO: size_t to int
        //li(t1, static_cast<int>(jep.work_amount));

        // TODO: loop is not unrolled
        L(unroll_loop_label);
        {
            const size_t loop_step = min_src_size;
            // TODO: use register, made it once
            const auto reg_loop_step = t5;
            li(reg_loop_step, loop_step);


//            const size_t vlen = 16;
//            const size_t vec_step = vlen / exec_prc.size();
//             if (print_tensors) {
//                 std::cout << "min_src_size: " << min_src_size << std::endl;
//                 std::cout << "vec_step: " << vec_step << std::endl;
//             }

            blt(reg_work_amount, reg_loop_step, unroll_loop_end_label);

            // aux gpr is used by emitters
            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    add(get_aux_gpr_kernel(i), x0, get_src_reg(i));
                }
            }

            // TODO: size_t to int
            li(t1, static_cast<int>(min_src_size));

            L(unroll_loop_label2);
            {
                vsetvli(t0, t1, SEW::e32, lmul);

                sub(t1, t1, t0);
                slli(t0, t0, 2);

                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        //load_vector(get_vmm_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, false, j * vec_step * jep.src_prc[i].size());
                        vle32_v(get_vmm_reg(i, lmul), get_aux_gpr_kernel(i));
                        add(get_aux_gpr_kernel(i), get_aux_gpr_kernel(i), t0);
                    }
                }

                if (jit_compute) {
                    compute_eltwise_op(lmul, input_count, vmm_dst_idx);
                } else {
                    std::cout << "compute_eltwise_op is ignored" << std::endl;
                }

                apply_post_ops(lmul, input_count, vmm_dst_idx);

                vse32_v(get_dst_vmm(vmm_dst_idx, lmul), reg_dst);
                // TODO: don't change destination ptr
                add(reg_dst, reg_dst, t0);

                if (jit_loop) {
                    bnez(t1, unroll_loop_label2);
                } else {
                    std::cout << "unroll loop2 is ignored" << std::endl;
                }
            }

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] == jep.dst_size) {
//                    li(get_aux_gpr_kernel(i), jep.src_prc[i].size() * min_src_size);
//                    add(get_src_reg(i), get_src_reg(i), get_aux_gpr_kernel(i));

                    li(reg_tmp, jep.src_prc[i].size() * min_src_size);
                    add(get_src_reg(i), get_src_reg(i), reg_tmp);
                }
            }

            sub(reg_work_amount, reg_work_amount, reg_loop_step);
            if (jit_loop) {
                j_(unroll_loop_label);
            } else {
                std::cout << "unroll loop is ignored" << std::endl;
            }
        }

        L(unroll_loop_end_label);
    }

    if (min_src_size == jep.dst_size) {
//        if (print_tensors) {
//            std::cout << "jit_uni_eltwise_generic: loop" << std::endl;
//        }

        L(main_loop_label);
        {
            vsetvli(t0, reg_work_amount, SEW::e32, lmul);
            sub(reg_work_amount, reg_work_amount, t0);
            slli(t0, t0, 2);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    std::cout << "input " << i << ": vector" << std::endl;
                    vle32_v(get_vmm_reg(i, lmul), get_src_reg(i));

                    add(get_src_reg(i), get_src_reg(i), t0);
                }
            }

            if (jit_compute) {
                compute_eltwise_op(lmul, input_count, vmm_dst_idx);
            }

            apply_post_ops(lmul, input_count, vmm_dst_idx);

            vse32_v(get_dst_vmm(vmm_dst_idx, lmul), reg_dst);
            // TODO: can we change reg_dst here?
            add(reg_dst, reg_dst, t0);

            if (jit_loop) {
                bnez(reg_work_amount, main_loop_label);
            } else {
                std::cout << "main loop is ignored" << std::endl;
            }
        }
        L(main_loop_end_label);
    }

    postamble();

    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        post_op_emitters[i]->emit_data();
    }
}

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    ov::intel_cpu::riscv64::jit_generator *host;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.exec_prc);
    }
};

//template<>
//struct EltwiseEmitter<jit_power_static_emitter2> {
//    void operator()(EltwiseEmitterContext& ctx) {
//        ctx.emitter = std::make_shared<jit_power_static_emitter2>(ctx.host,
//                                                                  ctx.opData.alpha,
//                                                                  ctx.opData.beta,
//                                                                  ctx.opData.gamma,
//                                                                  ctx.exec_prc);
//    }
//};

template<>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host,
                                                                 ctx.opData.alpha,
                                                                 ctx.opData.beta,
                                                                 ctx.opData.gamma,
                                                                 ctx.exec_prc);
    }
};

std::shared_ptr<jit_emitter> jit_uni_eltwise_generic::create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {
        nullptr,
        this,
        data,
        exec_prec
    };

    OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
    OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
    OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
    OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
    //OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
    OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
    OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));

    if (!ctx.emitter)
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");

    return ctx.emitter;
}

void jit_uni_eltwise_generic::compute_eltwise_op(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx) {
    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_count(); i++) {
        in_idxs.push_back(get_vmm_reg(i, lmul).getIdx());
    }

    std::vector<size_t> aux_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_vecs_count(); i++) {
        aux_idxs.push_back(get_aux_vmm(i, lmul, input_reg_count).getIdx());
    }

    std::vector<size_t> out_idxs;
    out_idxs.push_back(get_dst_vmm(vmm_dst_idx, lmul).getIdx());

    std::vector<size_t> gpr_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_gprs_count(); i++) {
        gpr_idxs.push_back(get_aux_gpr(i).getIdx());
    }

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs, gpr_idxs);
}

void jit_uni_eltwise_generic::apply_post_ops(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx) {
    int input_idx = eltwise_emitter->get_inputs_count();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < ops_list_.size(); i++) {
        if (ops_list_[i] == ov::intel_cpu::Type::Eltwise) {
            std::vector<size_t> in_idxs;
            in_idxs.push_back(get_dst_vmm(vmm_dst_idx, lmul).getIdx());
            for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_count(); j++)
                in_idxs.push_back(get_vmm_reg(input_idx++, lmul).getIdx());

            std::vector<size_t> out_idxs;
            out_idxs.push_back(get_vmm_reg(vmm_dst_idx, lmul).getIdx());

            std::vector<size_t> aux_vmm_idxs;
            for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->get_aux_vecs_count(); j++)
                aux_vmm_idxs.push_back(get_aux_vmm(j, lmul, input_reg_count).getIdx());

            std::vector<size_t> aux_gpr_idxs;
            for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->get_aux_gprs_count(); j++)
                aux_gpr_idxs.push_back(get_aux_gpr(j).getIdx());

            post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_vmm_idxs, aux_gpr_idxs);

            eltwise_post_op_idx++;
        } else if (ops_list_[i] == ov::intel_cpu::Type::FakeQuantize) {
            OPENVINO_THROW("Eltwise jit kernel: FakeQuantize is not supported");
        } else {
            OPENVINO_THROW("Eltwise jit kernel: unexpected operation type");
        }
    }
}

uint32_t jit_uni_eltwise_generic::lmul2int(const LMUL lmul) {
    switch (lmul) {
        case LMUL::m1: {
            return 1;
        }
        case LMUL::m2: {
            return 2;
        }
        case LMUL::m4: {
            return 4;
        }
        case LMUL::m8: {
            return 8;
        }
        default: {
            OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(static_cast<uint32_t>(lmul)));
        }
    }
}

Reg jit_uni_eltwise_generic::get_src_reg(uint32_t idx) {
    if (idx > MAX_ELTWISE_INPUTS) {
        OPENVINO_THROW("source vector ptr register " + std::to_string(idx) + " is not supported");
    }
    return Reg(18 + idx);
}

Reg jit_uni_eltwise_generic::get_aux_gpr(const uint32_t idx) {
    const uint32_t begin_idx = 28;
    const uint32_t end_idx = 31;
    if ((begin_idx + idx) > end_idx) {
        OPENVINO_THROW("aux gpr register " + std::to_string(idx) + " is not supported");
    }

    return Reg(begin_idx + idx);
}

Reg jit_uni_eltwise_generic::get_aux_gpr_kernel(const uint32_t idx) {
    const uint32_t begin_idx = 24;
    const uint32_t end_idx = 26;
    if ((begin_idx + idx) > end_idx) {
        OPENVINO_THROW("aux gpr register kernel " + std::to_string(idx) + " is not supported");
    }

    return Reg(begin_idx + idx);
}

VReg jit_uni_eltwise_generic::get_vmm_reg(const uint32_t idx, const LMUL lmul) {
    const uint32_t physical_idx = idx * lmul2int(lmul);
    if (physical_idx > 31) {
        OPENVINO_THROW("source vector register " + std::to_string(idx) + " (" + std::to_string(physical_idx) + ") is not supported");
    }
    return VReg(physical_idx);
}

VReg jit_uni_eltwise_generic::get_dst_vmm(const uint32_t idx, const LMUL lmul) {
    const uint32_t physical_idx = idx * lmul2int(lmul);
    if (physical_idx > 31) {
        OPENVINO_THROW("destination vector register " + std::to_string(idx) + " (" + std::to_string(physical_idx) + ") is not supported");
    }
    return VReg(physical_idx);
}

//SReg jit_uni_eltwise_generic::get_scl_reg(const uint32_t idx) {
//    if (idx > MAX_ELTWISE_INPUTS) {
//        OPENVINO_THROW("source scalar register " + std::to_string(idx) + " is not supported");
//    }
//    return SReg(0 + idx);
//}

VReg jit_uni_eltwise_generic::get_aux_vmm(const uint32_t idx, const LMUL lmul, const uint32_t start_idx) {
    const uint32_t phisical_idx =(start_idx + idx) * lmul2int(lmul);
    if (phisical_idx > 31) {
        OPENVINO_THROW("aux vector register " + std::to_string(idx) + " is not supported");
    }
    return VReg(phisical_idx);
}

void jit_uni_eltwise_generic::load_vector(const VReg& data,
                                          const Reg& ptr_reg,
                                          const ov::element::Type& src_prc,
                                          const ov::element::Type& dst_prc,
                                          const bool broadcast,
                                          const int32_t ptr_offset) {
}

void jit_uni_eltwise_generic::store_vector(const Reg& ptr,
                                           const VReg& data,
                                           const ov::element::Type& src_prc,
                                           const ov::element::Type& dst_prc,
                                           const int32_t ptr_offset) {
}

namespace {
template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<std::vector<element::Type>> &precisions) {
        precisions = T::get_supported_precisions();
    }
};
}

ov::element::Type eltwise_precision_helper::get_precision(const size_t inputs_number,
                                                          const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                                          const std::vector<EltwiseData>& eltwise_data) {
    ov::element::Type exec_prc = ov::element::undefined;
    return exec_prc;
}

std::set<std::vector<element::Type>> eltwise_precision_helper::get_supported_precisions(const Algorithm& algo) {
    std::set<std::vector<element::Type>> precisions;

    OV_SWITCH(intel_cpu, SupportedPrecisions, precisions, algo,
              OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
              OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
              OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
              //OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
              OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
              OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));
    if (precisions.empty())
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");

    return precisions;
}

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov

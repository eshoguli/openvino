// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "csinn/csinn_data_structure.h"
#include "csinn/csinn_runtime.h"
#include "memory_desc/cpu_memory_desc.h"


namespace ov {
namespace intel_cpu {

using csinn_tensor_ptr = std::shared_ptr<csinn_tensor>;
using csinn_session_ptr = std::shared_ptr<csinn_session>;

/**
* @brief Initializes shape in SHL tensor 
* @param dims vector of dimensions
* @param csinn_tensor_ptr shared pointer of SHL tensor
*/
inline void initShlTensorShape(const VectorDims& dims, const csinn_tensor_ptr& tensor) {
    tensor->dim_count = dims.size();
    OPENVINO_ASSERT(tensor->dim_count < MAX_DIM, "SHL supports shapes with rank less or equal to 8");
    for (int i = 0; i < tensor->dim_count; ++i)
        tensor->dim[i] = static_cast<int32_t>(dims[i]);
}

/**
* @brief Return SHL DataType that corresponds to the given precision
* @param precision precision to be converted
* @return SHL DataType
*/
inline csinn_dtype_enum precisionToShlDataType(ov::element::Type precision) {
    switch (precision) {
        case ov::element::i8:   return CSINN_DTYPE_INT8;
        case ov::element::u8:   return CSINN_DTYPE_UINT8;
        case ov::element::i16:  return CSINN_DTYPE_INT16;
        case ov::element::u16:  return CSINN_DTYPE_UINT16;
        case ov::element::i32:  return CSINN_DTYPE_INT32;
        case ov::element::u32:  return CSINN_DTYPE_UINT32;
        case ov::element::f16:  return CSINN_DTYPE_FLOAT16;
        case ov::element::f32:  return CSINN_DTYPE_FLOAT32;
        case ov::element::f64:  return CSINN_DTYPE_FLOAT64;
        case ov::element::i64:  return CSINN_DTYPE_INT64;
        case ov::element::bf16: return CSINN_DTYPE_BFLOAT16;
        default:
            OPENVINO_THROW("Unknown data type for SHL");
    }
}

/**
* @brief Return SHL DataLayout that corresponds to MemoryDecs layout
* @param desc MemoryDecs from which layout is retrieved
* @param is_weights True if it's a layout of weights
* @return SHL DataLayout or CSINN_LAYOUT_NULL if the layout is unknown
*/
inline csinn_layout_enum getShlDataLayoutByMemoryDesc(const MemoryDescPtr& desc, bool is_weights = false) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        switch (desc->getShape().getRank()) {
            case 1: return is_weights ? CSINN_LAYOUT_O     : CSINN_LAYOUT_N;
            case 2: return is_weights ? CSINN_LAYOUT_OI    : CSINN_LAYOUT_NC;
            case 3: return is_weights ? CSINN_LAYOUT_OIW   : CSINN_LAYOUT_NCW;
            case 4: return is_weights ? CSINN_LAYOUT_OIHW  : CSINN_LAYOUT_NCHW;
            case 5: return is_weights ? CSINN_LAYOUT_OIDHW : CSINN_LAYOUT_NCDHW;
        }
    } else if (desc->hasLayoutType(LayoutType::nspc)) {
        switch (desc->getShape().getRank()) {
            case 3: return is_weights ? CSINN_LAYOUT_OWI   : CSINN_LAYOUT_NWC;
            case 4: return is_weights ? CSINN_LAYOUT_OHWI  : CSINN_LAYOUT_NHWC;
            case 5: return is_weights ? CSINN_LAYOUT_ODHWI : CSINN_LAYOUT_NDHWC;
        }
    }
    return CSINN_LAYOUT_NULL;
}

/**
* @brief Allocate SHL session
* @return SHL session shared pointer
*/
inline csinn_session_ptr allocateShlSession() {
    const auto sess = std::shared_ptr<csinn_session>(csinn_alloc_session());
    OPENVINO_ASSERT(sess != nullptr, "Failed to create session");
    return sess;
}

/**
* @brief Allocate SHL params for specific OP
* @param sess shared pointer of SHL session
* @return SHL typed params shared pointer
*/
template<typename T>
inline std::shared_ptr<T> allocateShlParams(const csinn_session_ptr& sess) {
    const auto params = std::shared_ptr<T>(static_cast<T*>(csinn_alloc_params(sizeof(T), sess.get())));
    OPENVINO_ASSERT(params != nullptr, "Failed to allocate SHL params");
    return params;
}

/**
* @brief Allocate default SHL tensor
* @param sess shared pointer of SHL session
* @return SHL tensor shared pointer
*/
inline csinn_tensor_ptr allocateShlTensor(const csinn_session_ptr& sess) {
    auto tensor = std::shared_ptr<csinn_tensor>(csinn_alloc_tensor(sess.get()));
    OPENVINO_ASSERT(tensor != nullptr, "Failed to create SHL tensor");
    return tensor;
}

}   // namespace intel_cpu
}   // namespace ov

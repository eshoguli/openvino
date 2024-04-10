// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_fullyconnected.hpp"

#include "csinn/csi_nn.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using namespace executor;
using namespace ov::element;

bool ShlFCExecutor::supports(const FCConfig& config) {
    if (!config.postOps.empty()) {
        DEBUG_LOG("ShlFCExecutor: PostOps are not supported");
        return false;
    }

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);
    if (!everyone_is(ov::element::f32, srcDesc->getPrecision(), weiDesc->getPrecision(), dstDesc->getPrecision())) {
        DEBUG_LOG("ShlFCExecutor: supports only f32");
        return false;
    }

    if (config.attrs.withBias) {
        const auto& biaDesc = config.descs.at(ARG_BIAS);
        if (biaDesc->getPrecision() != ov::element::f32) {
            DEBUG_LOG("ShlFCExecutor: supports only f32 bias");
            return false;
        }

        const auto& biasDims = biaDesc->getShape().getStaticDims();
        const auto& outDims = dstDesc->getShape().getDims();
        const bool isByChannel = biasDims.back() == outDims.back();
        if (!isByChannel || !std::all_of(biasDims.begin(), biasDims.end() - 1, [](const Dim dim) { return dim == 1; })) {
            DEBUG_LOG("ShlFCExecutor: only 'by channel' bias is supported");
            return false;
        }
    }

    return true;
}

ShlFCExecutor::ShlFCExecutor(const FCAttrs& attrs,
                             const PostOps& postOps,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr context) {
    const auto& strDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    // Allocate SHL session
    sess = allocateShlSession();
    sess->base_run_mode = CSINN_RM_LAYER;

    // Allocate SHL tensors
    src = allocateShlTensor(sess);
    wei = allocateShlTensor(sess);
    dst = allocateShlTensor(sess);
    bias = allocateShlTensor(sess);

    // Init precisions
    src->dtype = precisionToShlDataType(strDesc->getPrecision());
    wei->dtype = precisionToShlDataType(weiDesc->getPrecision());
    dst->dtype = precisionToShlDataType(dstDesc->getPrecision());

    // Init layouts
    src->layout = getShlDataLayoutByMemoryDesc(strDesc, false);
    wei->layout = getShlDataLayoutByMemoryDesc(weiDesc, true);
    dst->layout = getShlDataLayoutByMemoryDesc(dstDesc, false);

    if (attrs.withBias) {
        const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
        bias->dtype = precisionToShlDataType(biasDesc->getPrecision());
        bias->layout = getShlDataLayoutByMemoryDesc(biasDesc);
        bias->data = memory.at(ARG_BIAS)->getData();
        initShlTensorShape(memory.at(ARG_BIAS)->getDescPtr()->getShape().getStaticDims(), bias);
    }

    // Init FC params
    params = allocateShlParams<csinn_fc_params>(sess);
    params->base.api = CSINN_RVV;

    int status = csinn_fullyconnected_init(src.get(), dst.get(), wei.get(), bias.get(), params.get());
    OPENVINO_ASSERT(status > 0, "ShlFCExecutor: failed to init FC");
}

bool ShlFCExecutor::update(const MemoryArgs& memory) {
    initShlTensorShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims(), src);
    initShlTensorShape(memory.at(ARG_WEI)->getDescPtr()->getShape().getStaticDims(), wei);
    initShlTensorShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims(), dst);
    return true;
}

void ShlFCExecutor::execute(const MemoryArgs& memory) {
    src->data = memory.at(ARG_SRC)->getData();
    wei->data = memory.at(ARG_WEI)->getData();
    dst->data = memory.at(ARG_DST)->getData();

    int status = csinn_fullyconnected(src.get(), dst.get(), wei.get(), bias.get(), params.get());
    OPENVINO_ASSERT(status > 0, "ShlFCExecutor: failed to execute");
}

}  // namespace intel_cpu
}  // namespace ov

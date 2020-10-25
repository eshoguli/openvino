// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <inference_engine.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>
#include <gna/gna_config.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <legacy/graph_tools.hpp>
#include <legacy/details/ie_cnn_network_tools.h>

#include "blob_factory.hpp"
#include "blob_dumper.hpp"
#include "blob_dump.h"

using namespace InferenceEngine;

class OperationName {
public:
    enum sortType {
        byName,
        byInference
    };

    OperationName(const OperationName::sortType sort) : sort(sort), executionIndex(0ul) {}

    std::string getName(const std::string& operationType) {
        const auto it = operationIndexes.find(operationType);

        size_t index = 0ul;
        if (it == operationIndexes.end()) {
            operationIndexes[operationType] = 0;
            index = 0ul;
        }
        else {
            index = it->second;
            operationIndexes[operationType] = ++index;
        }

        return (sort == OperationName::sortType::byName) ?
            operationType + "_" + std::to_string(index) :
            std::to_string(executionIndex++) + "_" + operationType + "_" + std::to_string(index);
    }

private:
    std::map<std::string, size_t> operationIndexes;
    size_t executionIndex;
    OperationName::sortType sort;
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_model.empty()) {
        throw std::logic_error("Model is required but not set. Please set -model option.");
    }

    if (FLAGS_dump_dir.empty()) {
        throw std::logic_error("Dump directory is required but not set. Please set -dump_dir option.");
    }

    return true;
}

template<typename T>
bool isScalarLike(const Blob::Ptr& blob) {
    const auto* buffer = blob->buffer().as<T*>();
    const size_t blobSize = blob->size();
    for (size_t i = 1; i < blobSize; i++) {
        if (buffer[i] != buffer[0]) {
            return false;
        }
    }
    return true;
}

bool isScalarLike(const Blob::Ptr& blob) {
    size_t data_size = blob->size();
    if (data_size == 1ul) {
        return true;
    }

    const auto precision = blob->getTensorDesc().getPrecision();
    switch (precision) {
    case Precision::FP32: {
        if (!isScalarLike<float>(blob)) {
            return false;
        }
        break;
    }
    case Precision::I32: {
        if (!isScalarLike<int32_t>(blob)) {
            return false;
        }
        break;
    }
    case Precision::BF16:
    case Precision::I16: {
        if (!isScalarLike<int16_t>(blob)) {
            return false;
        }
        break;
    }
    case Precision::I64: {
        if (!isScalarLike<int64_t>(blob)) {
            return false;
        }
        break;
    }
    case Precision::U16: {
        if (!isScalarLike<uint16_t>(blob)) {
            return false;
        }
        break;
    }
    case Precision::I8: {
        if (!isScalarLike<int8_t>(blob)) {
            return false;
        }
        break;
    }
    case Precision::U8: {
        if (!isScalarLike<uint8_t>(blob)) {
            return false;
        }
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Dumper. Unsupported precision";
    }

    return true;
}

template<typename T>
void setFirstValue(const Blob::Ptr& originalBlob, Blob::Ptr& resultBlob) {
    const auto *originalBuffer = originalBlob->buffer().as<T*>();
    auto *resultBuffer = resultBlob->buffer().as<T*>();
    resultBuffer[0] = originalBuffer[0];
}

void setValue(const Blob::Ptr& originalBlob, Blob::Ptr& resultBlob) {
    const auto precision = resultBlob->getTensorDesc().getPrecision();
    switch (precision) {
    case Precision::FP32: {
        setFirstValue<float>(originalBlob, resultBlob);
        break;
    }
    case Precision::I32: {
        setFirstValue<int32_t>(originalBlob, resultBlob);
        break;
    }
    case Precision::BF16:
    case Precision::I16: {
        setFirstValue<int16_t>(originalBlob, resultBlob);
        break;
    }
    case Precision::I64: {
        setFirstValue<int64_t>(originalBlob, resultBlob);
        break;
    }
    case Precision::U16: {
        setFirstValue<uint16_t>(originalBlob, resultBlob);
        break;
    }
    case Precision::I8: {
        setFirstValue<int8_t>(originalBlob, resultBlob);
        break;
    }
    case Precision::U8: {
        setFirstValue<uint8_t>(originalBlob, resultBlob);
        break;
    }
    default:
        THROW_IE_EXCEPTION << "Dumper. Unsupported precision";
    }
}

Blob::Ptr reduce(const Blob::Ptr& originalBlob) {
    if (!isScalarLike(originalBlob)) {
        return originalBlob;
    }

    const auto desc = originalBlob->getTensorDesc();
    Blob::Ptr newBlob = make_plain_blob(desc.getPrecision(), {});
    newBlob->allocate();
    setValue(originalBlob, newBlob);
    return newBlob;
}

/**
* @brief The entry point of the blob_dumper application
*/
int main(int argc, char *argv[]) {
    try {
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(FLAGS_model);

        IE_SUPPRESS_DEPRECATED_START
        std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(cnnNetwork);
        IE_SUPPRESS_DEPRECATED_END

        auto getInsDataIndex = [](const CNNLayerPtr& child, const CNNLayerPtr& parent) -> int {
            for (size_t i = 0; i < child->insData.size(); ++i) {
                const CNNLayerPtr layer = getCreatorLayer(child->insData[i].lock()).lock();
                if (layer.get() == parent.get()) {
                    return i;
                }
            }

            THROW_IE_EXCEPTION << "not found";
        };

        OperationName operationName(OperationName::byInference);

        for (auto layer : layers) {
            if (FLAGS_ignore_const && (layer->type == "Const")) {
                continue;
            }

            if (!layer->blobs.empty()) {
                for (auto blobIt : layer->blobs) {
                    CNNLayerPtr childLayerToBuildName;
                    std::string nodeName = layer->name;
                    if (layer->type == "Const") {
                        //if (layer->name == "564/add__Const_564/fq_input_0_ScaleShift_564/add_") {
                        //    std::cout << "";
                        //}
                        const std::map<std::string, CNNLayerPtr>& inputTo = getInputTo(layer->outData[0]);
                        childLayerToBuildName = (inputTo.size() == 1ul) ? inputTo.begin()->second : nullptr;
                        if (childLayerToBuildName == nullptr) {
                            CNNLayerPtr fakeQuantize = nullptr;
                            for (const auto it : inputTo) {
                                const auto childLayer = it.second;
                                if (childLayer->type != "FakeQuantize") {
                                    fakeQuantize = nullptr;
                                    break;
                                }

                                if (fakeQuantize == nullptr) {
                                    fakeQuantize = childLayer;
                                }
                                else if (childLayer.get() == fakeQuantize.get()) {
                                    break;
                                }
                            }

                            childLayerToBuildName = inputTo.begin()->second;
                        }

                        if (childLayerToBuildName == nullptr) {
                            nodeName = layer->name;
                        }
                        else {
                            int index = getInsDataIndex(childLayerToBuildName, layer);
                            if (childLayerToBuildName->type == "Eltwise") {
                                const std::map<std::string, CNNLayerPtr>& childInputTo = getInputTo(childLayerToBuildName->outData[0]);
                                const auto conv = (childInputTo.size() == 1ul) ? childInputTo.begin()->second : nullptr;
                                if (conv->type == "Convolution") {
                                    nodeName = conv->name + "_eltwise";
                                }
                                else {
                                    nodeName = childLayerToBuildName->name + "_const_input_" + std::to_string(index);
                                }
                            }
                            else {
                                nodeName = childLayerToBuildName->name + "_const_input_" + std::to_string(index);
                            }
                        }
                    }
                    else if (layer->type == "ScaleShift") {
                        const std::map<std::string, CNNLayerPtr>& inputTo = getInputTo(layer->outData[0]);
                        childLayerToBuildName = (inputTo.size() == 1ul) ? inputTo.begin()->second : nullptr;
                        if (childLayerToBuildName != nullptr) {
                            int index = getInsDataIndex(childLayerToBuildName, layer);
                            nodeName = childLayerToBuildName->name + "_scaleshift_input_" + std::to_string(index);
                        }
                    }

                    std::stringstream ss;
                    for (size_t i = 0; i < layer->outData.size(); ++i) {
                        ss << "_" << layer->outData[i]->getPrecision();
                    }
                    nodeName += ss.str();

                    // std::cout << layer->name << ": " << layer->type << std::endl;

                    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
                    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
                    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
                    std::replace(nodeName.begin(), nodeName.end(), ':', '-');

                    if ((layer->type == "FakeQuantize") || ((childLayerToBuildName != nullptr) && (childLayerToBuildName->type == "FakeQuantize"))) {
                        const size_t index = nodeName.find("_original", 0);
                        if (index != -1) {
                            nodeName.replace(index, std::string("_original").size(), "");
                        }
                    }

                    if (FLAGS_rename) {
                        nodeName = operationName.getName(layer->type);
                    }

                    const std::string blobName = blobIt.first;
                    const std::string dumpFilePath = FLAGS_dump_dir + "\\" + nodeName + "_" + blobName + ".ieb";

                    Blob::Ptr blob = blobIt.second;
                    if (FLAGS_reduce) {
                        blob = reduce(blob);
                    }

                    BlobDumper dumper(blob);
                    dumper.dumpAsTxt(dumpFilePath);
                }
            } else {
                auto nodeName = layer->name;
                std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
                std::replace(nodeName.begin(), nodeName.end(), '/', '_');
                std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
                std::replace(nodeName.begin(), nodeName.end(), ':', '-');

                std::stringstream ss;
                for (size_t i = 0; i < layer->outData.size(); ++i) {
                    ss << "_" << layer->outData[i]->getPrecision();
                }
                nodeName += ss.str();

                if (FLAGS_rename) {
                    nodeName = operationName.getName(layer->type);
                }

                const std::string dumpFilePath = FLAGS_dump_dir + "\\" + nodeName + ".ieb";

                BlobDumper dumper(nullptr);
                dumper.dumpAsTxt(dumpFilePath);
            }
        }
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 3;
    }

    return 0;
}

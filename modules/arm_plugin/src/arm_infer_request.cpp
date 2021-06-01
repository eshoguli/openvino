// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <map>

#include <ie_blob.h>
#include <debug.h>
#include <ie_layouts.h>
#include <threading/ie_executor_manager.hpp>
#include <blob_transform.hpp>
#include <precision_utils.h>
#include <ngraph/function.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/runtime/reference/convert.hpp>

#include "arm_infer_request.hpp"
#include "arm_executable_network.hpp"
#include "arm_plugin.hpp"


using namespace ArmPlugin;
using namespace InferenceEngine;
using namespace openvino;

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using fsec = std::chrono::duration<float>;

ArmInferRequest::ArmInferRequest(const InferenceEngine::InputsDataMap&                networkInputs,
                                 const InferenceEngine::OutputsDataMap&               networkOutputs,
                                 const std::shared_ptr<ArmPlugin::ExecutableNetwork>& executableNetwork) :
    IInferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork),
#if 1
    _lifetime{std::make_shared<arm_compute::OffsetLifetimeManager>()},
#else
    _lifetime{std::make_shared<arm_compute::BlobLifetimeManager>()},
#endif
    _pool{std::make_shared<arm_compute::PoolManager>()},
    _memoryManager{std::make_shared<arm_compute::MemoryManagerOnDemand>(_lifetime, _pool)},
    _memoryGroup{_memoryManager} {
    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));
    Layer::Map layers;
    IE_ASSERT(_executableNetwork->_executor != nullptr);
    _executableNetwork->_executor->runAndWait({
        [&] {
            layers = Converter{_executableNetwork->_function}.Configure(_memoryManager, _memoryGroup);
        }
    });
    auto allocateBlob = [] (const auto& blobData, auto& blob, auto tensor, auto output) {
        auto networkPresion = InferenceEngine::details::convertPrecision(output.get_element_type());
        InferenceEngine::Blob::Ptr networkBlob;
        if (ngraph::op::is_constant(output.get_node())) {
            if (networkPresion == blobData->getTensorDesc().getPrecision()) {
                networkBlob = blob = make_blob_with_precision(blobData->getTensorDesc(),
                                                              static_cast<arm_compute::Tensor*>(tensor)->buffer());
            } else {
                blob = make_blob_with_precision(blobData->getTensorDesc());
                blob->allocate();
                networkBlob = make_blob_with_precision({networkPresion,
                                                        blobData->getTensorDesc().getDims(),
                                                        blobData->getTensorDesc().getLayout()},
                                                        static_cast<arm_compute::Tensor*>(tensor)->buffer());
            }
        } else {
            blob = make_blob_with_precision(blobData->getTensorDesc());
            blob->allocate();
            if (networkPresion == blobData->getTensorDesc().getPrecision()) {
                networkBlob = blob;
            } else {
                networkBlob = make_blob_with_precision({networkPresion,
                                                        blobData->getTensorDesc().getDims(),
                                                        blobData->getTensorDesc().getLayout()});
                networkBlob->allocate();
            }
        }
        return networkBlob;
    };
    for (auto&& node : _executableNetwork->_function->get_parameters()) {
        auto nodeName = node->get_friendly_name();
        IE_ASSERT(node->outputs().size() == 1);
        for (auto&& output : node->outputs()) {
            auto& tensor = layers.at(node->get_instance_id())._outputs.at(output);
            if (tensor->info()->has_padding()) {
                tensor->allocator()->allocate();
            }
            _inputInfo.emplace_back(IOInfo{
                output,
                tensor.get(),
                openvino::itt::handle(_executableNetwork->_function->get_friendly_name() + "_" +
                                      requestID + "_preprocessing_" +
                                      node->get_friendly_name() + "_" +
                                      std::to_string(node->get_instance_id())),
                allocateBlob(_networkInputs.at(nodeName),
                             _inputs[nodeName],
                             tensor.get(),
                             output),
                _inputs.find(nodeName)});
        }
    }
    for (auto&& node : _executableNetwork->_function->get_results()) {
        IE_ASSERT(node->inputs().size() == 1);
        auto outputName = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(
            node->get_rt_info().at("ResultName"))->get();
        IE_ASSERT(details::contains(_networkOutputs, outputName)) << outputName;
        auto input = node->input(0);
        auto sourceOutput = input.get_source_output();
        auto& tensor = layers.at(node->get_instance_id())._inputs.at(input);
        if (tensor->info()->has_padding()) {
            tensor->allocator()->allocate();
        }
        _outputInfo.emplace_back(IOInfo{
            sourceOutput,
            tensor,
            openvino::itt::handle(_executableNetwork->_function->get_friendly_name() + "_" +
                                requestID + "_postprocessing_" +
                                outputName + "_" +
                                std::to_string(node->get_instance_id())),
            allocateBlob(_networkOutputs.at(outputName),
                        _outputs[outputName],
                        tensor,
                        sourceOutput),
            _outputs.find(outputName)});
    }
    IE_ASSERT(!_outputInfo.empty());
    _memoryManager->populate(_allocator, 1);
    _memoryGroupScope = std::make_shared<arm_compute::MemoryGroupResourceScope>(_memoryGroup);
    for (auto&& node : _executableNetwork->_function->get_ordered_ops()) {
        if (ngraph::is_type<opset::ArmNoOp>(node.get())) {
            IE_ASSERT(node->inputs().size() == 1);
            IE_ASSERT(node->outputs().size() == 1);
            auto& layer = layers.at(node->get_instance_id());
            layer._outputs.at(node->output(0))->allocator()->import_memory(
                layer._inputs.at(node->input(0))->buffer());
        }
    }
    for (auto&& node : _executableNetwork->_function->get_ordered_ops()) {
        _layers.emplace_back(LayerInfo{
            std::move(layers.at(node->get_instance_id())),
            node.get(),
            openvino::itt::handle(_executableNetwork->_function->get_friendly_name() + "_" +
                                  requestID + "_" +
                                  node->get_friendly_name() + "_" +
                                  std::to_string(node->get_instance_id()))});
    }
}

ArmInferRequest::~ArmInferRequest() {
    _executableNetwork->_requestId--;
}

struct StaticCast {
    template<typename T> operator T*() {return static_cast<T*>(_ptr);}
    void* _ptr;
};
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto apply = [&] (auto convert) {
        convert(
            StaticCast{InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<void*>()},
            StaticCast{InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<void*>()},
            src->size());
    };

    CallSwitch(
        AP_WRAP(apply, ngraph::runtime::reference::convert),
        InferenceEngine::details::convertPrecision(src->getTensorDesc().getPrecision()), merge(allTypes, boolType),
        InferenceEngine::details::convertPrecision(dst->getTensorDesc().getPrecision()), allTypes);
}

void ArmInferRequest::InferImpl() {
    {
        execDataPreprocessing(_inputs);
        for (auto&& input : _inputInfo) {
            auto start = Time::now();
            OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, input._profilingTask);
            const auto& inputBlob = input._itBlob->second;
            if (inputBlob != input._blob) {
                if (input._blob->getTensorDesc() == inputBlob->getTensorDesc()) {
                    input._blob = inputBlob;
                } else {
                    blobCopy(inputBlob, input._blob);
                }
            }
            if (input._tensor->info()->has_padding()) {
                arm_compute::Tensor inputTensor;
                inputTensor.allocator()->init({input._tensor->info()->tensor_shape(), 1, input._tensor->info()->data_type()});
                inputTensor.allocator()->import_memory(
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(input._blob)->rmap().as<void*>());
                input._tensor->copy_from(inputTensor);
            } else {
                static_cast<arm_compute::Tensor*>(input._tensor)->allocator()->import_memory(
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(input._blob)->rmap().as<void*>());
            }
            input._duration += Time::now() - start;
            input._counter++;
        }
        for (auto&& output : _outputInfo) {
            const auto& outputBlob = output._itBlob->second;
            if (!ngraph::op::is_constant(output._output.get_node())) {
                if (outputBlob != output._blob) {
                    if (output._blob->getTensorDesc() == outputBlob->getTensorDesc()) {
                        output._blob = outputBlob;
                    }
                }
                if (!output._tensor->info()->has_padding()) {
                    static_cast<arm_compute::Tensor*>(output._tensor)->allocator()->import_memory(
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(output._blob)->wmap().as<void*>());
                }
            }
        }
    }
    {
        for (auto&& layer : _layers) {
            if (layer._layer._function != nullptr) {
                OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, layer._profilingTask);
                auto start = Time::now();
                layer._layer._function->run();
                layer._duration += Time::now() - start;
                layer._counter++;
            }
        }
    }
    {
        for (auto&& output : _outputInfo) {
            auto start = Time::now();
            OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, output._profilingTask);
            const auto& outputBlob = output._itBlob->second;
            if (ngraph::op::is_constant(output._output.get_node())) {
                if (outputBlob != output._blob) {
                    blobCopy(output._blob, outputBlob);
                }
            } else {
                if (output._tensor->info()->has_padding()) {
                    arm_compute::Tensor outputTensor;
                    outputTensor.allocator()->init({output._tensor->info()->tensor_shape(), 1, output._tensor->info()->data_type()});
                    outputTensor.allocator()->import_memory(
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(output._blob)->wmap().as<void*>());
                    outputTensor.copy_from(*(output._tensor));
                }
                if (outputBlob != output._blob) {
                    if (output._blob->getTensorDesc() != outputBlob->getTensorDesc()) {
                        blobCopy(output._blob, outputBlob);
                    }
                }
            }
            output._duration += Time::now() - start;
            output._counter++;
        }
    }
}

std::map<std::string, InferenceEngineProfileInfo> ArmInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    int executionIndex = 0;
    auto fillInfo = [&] (const auto& layer, ngraph::Node* node, auto& name) {
        InferenceEngineProfileInfo info;
        info.execution_index = executionIndex;
        ++executionIndex;
        info.status = InferenceEngineProfileInfo::EXECUTED;
        info.cpu_uSec = 0;
        info.realTime_uSec = layer._duration.count() / layer._counter;
        auto type = std::string {node->get_type_name()}
            + '.' + std::to_string(node->get_type_info().version);
        auto pos = std::copy_n(type.c_str(), std::min(sizeof(info.layer_type) - 1, type.size()), info.layer_type);
        *pos = '\0';
        perfMap.emplace(name, info);
    };
    for (auto&& input : _inputInfo) {
        fillInfo(input, input._output.get_node(), input._itBlob->first);
    }
    for (auto&& layer : _layers) {
        if (layer._layer._function != nullptr) {
            fillInfo(layer, layer._node, layer._node->get_friendly_name());
        }
    }
    for (auto&& output : _outputInfo) {
        fillInfo(output, output._output.get_node(), output._itBlob->first);
    }
    return perfMap;
}

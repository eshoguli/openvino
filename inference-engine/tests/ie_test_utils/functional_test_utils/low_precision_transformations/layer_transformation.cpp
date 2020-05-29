// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

namespace LayerTestsUtils {

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::I8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8U8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::U8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsI8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::I8 },
        { InferenceEngine::Precision::I8 });
}

InferenceEngine::details::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformer(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
    return transformer;
}

InferenceEngine::CNNNetwork LayerTransformation::transform(InferenceEngine::details::LayerTransformation::Params& params) {
    auto net1 = InferenceEngine::CNNNetwork(function);
    std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(net1);

    if (clonedNetwork->getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset2::SpaceToBatch>(node) ||
                std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node);
        };
        auto nGraphFunc = clonedNetwork->getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::CommonOptimizations(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet3ToOpSet2(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet2ToOpSet1(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy(transformations_callback).run_on_function(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);
    }

    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(clonedNetwork);
    if (cnnNetworkImp) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        InferenceEngine::ConstTransformer transformator(cnnNetworkImp.get());
        transformator.fullTrim();
    }

    auto transformer = getLowPrecisionTransformer(params);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

InferenceEngine::CNNNetwork LayerTransformation::transform(const InferenceEngine::details::LowPrecisionTransformations& transformations) {
    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

    InferenceEngine::details::LowPrecisionTransformer transformer(transformations);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

void LayerTransformation::checkParentPrecision(const InferenceEngine::CNNLayerPtr& layer, const bool lowPrecision) {
    EXPECT_EQ(1ul, layer->insData.size()) << "insert data count is no expected: " << layer->insData.size();
    const InferenceEngine::DataPtr insData = layer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr) << "insert data is nullable";
    const InferenceEngine::Precision precision = insData->getTensorDesc().getPrecision();

    const std::unordered_set<uint8_t> expectedPrecisions = lowPrecision ?
        std::unordered_set<uint8_t>({ InferenceEngine::Precision::U8, InferenceEngine::Precision::I8 }) :
        std::unordered_set<uint8_t>({ InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32 });
    EXPECT_TRUE((expectedPrecisions.find(precision) != expectedPrecisions.end())) <<
        "actual precision is " << precision;
}

std::string LayerTransformation::toString(const InferenceEngine::details::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric" : "symmetric") << "_" <<
        params.precisionsOnActivations << "_" <<
        params.precisionsOnWeights << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

}  // namespace LayerTestsUtils

# OpenVINO™ Low Precision Transformations

## Table of Contents

1. [Introduction](#introduction)  
2. [Input model requirements](#input-model-requirements)  
3. [Low precision tools](#low-precision-tools)  
4. [Quantization approaches](#quantization-approaches)  
   4.1. [FakeQuantize operation](#fakequantize-operation)  
   4.2. [Quantize and dequantization operations](#quantize-and-dequantization-operations)  
5. [Low precision transformations pipeline](#low-precision-transformations-pipeline)  
      5.1. [Step #1. Prerequisites](#step-1-prerequisites)  
      5.2. [Step #2. Markup](#step-2-markup)  
      5.3. [Step #3. Main transformations: FakeQuantize decomposition and dequantization operations handling](#step-3-main-transformations-fakequantize-decomposition-and-dequantization-operations-handling)  
      5.4. [Step #4. Cleanup result model](#step-4-cleanup-result-model)  
6. [Low precision transformations in plugin transformation pipeline](#low-precision-transformations-in-plugin-transformation-pipeline)  
      [Step #1. Common optimizations](#step-1-common-optimizations)  
      [Step #2. Low precision transformations execution](#step-2-low-precision-transformations-execution)  
      [Step #3. Plugin specific transformations](#step-3-plugin-specific-transformations)
7. [Result model overview](#result-model-overview)
8. [Mixed precision](#mixed-precision)
9. [Customization](#customization)
10. [Transformations](#transformations)  
    10.1. [Branch specific transformations](#branch-specific-transformations)  
    10.2. [Decomposition transformations](#decomposition-transformations)  
    10.3. [Main transformations](#main-transformations)  
    10.4. [Clean up transformations](#clean-up-transformations)  

## Introduction
The goal of `Low Precision Transformations` (LPT transformations) is transform quantized model from original precisions (FP16 or FP32) to low precision (INT8: `signed int8` or `unsigned int8`) model to prepare model for low precision inference in OpenVINO™ plugin. It achieved by two main principles:
1. `FakeQuantize` operation decomposition to two parts (you can explore details below in [Low precision transformations pipeline, step #2: decomposition](#step-2-decomposition) section):  
    - part #1: quantize operation - new `FakeQuantize` operation with output quantization intervals in low precision range (signed int8: [-128, 127] or [-127, 127], unsigned int8: [0, 255] or [0, 256]) and with low precision output (`signed int8` or `unsigned int8`), 
    - part #2: dequantization operations with low precision input and original precision output.
2. Dequantization operation propagation through original models operations to avoid dequantization operations before original model operations, thus the quantize operations with low precision output remains before original model operations. 

As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

How quantize a model in details you can explore in [Low precision tools](#low-precision-tools) section below. For more information about model quantization, refer to **Brief History of Lower Precision in Deep Learning** section in [this whitepaper](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training).

## Input model requirements

LPT transformations decompose `FakeQuantize` operations if `level` parameter is set to 255 or 256. LPT transformations propagate dequantization operations through follow operations:

| Operation        |
|------------------|
| Add              |
| Avg              |
| Clamp            |
| Concat           |
| Convolution      |
| DepthToSpace     |
| FakeQuantize     |
| GroupConvolution |
| Interpolate      |
| MatMul           |
| MaxPool          |
| Multiply         |
| MVN              |
| NormalizeL2      |
| PRelu            |
| Relu             |
| Reshape          |
| Split            |
| Squeeze          |
| StridedSlice     |
| Transpose        |
| Unsqueeze        |
| VariadicSplit    |

If operation is not supported by LPT then dequantization operation will be not propagated, input tensor precisions will be not changed to low precision and operation will be executed in original precision. 

For example, if you would like to infer `Convolution` operation in low precision then your model can look as on picture below:

![Quantized Convolution](convolution/img/convolution.actual.png)

> There are several supported quantization approaches on activations and on weights. All supported approaches are described in [Quantization approaches](#quantization-approaches) section below. In demonstrated model [Quantization approaches: FakeQuantize operation](#fakequantize-operation) approach is used.

### Low precision tools
There are two tools to quantize a model:
1. [Post-Training Optimization Toolkit](@ref pot_README.html) (POT)
2. [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf) (NNCF)

Additionally, low precision transformations can handle ONNX quantized model.

## Quantization approaches
LPT transformations support two quantization approaches:
1. `FakeQuantize` operation,
2. Quantize and dequantization operations

Let's explore both approaches in details on `Convolution` operation.
### FakeQuantize operation  
In this case `FakeQuantize` operation is used on activations and weights to quantize and dequantize `Convolution` input tensors. All operations are in original precision. Original input model:  
![](img/fq_and_convolution.common.png)  
In LPT result model you can see, that:
1. `FakeQuantize` operations on activations and on weights were decomposed to two operations: 
   - new  `FakeQuantize`operation with updated output intervals (on activations interval is [0, 255], on weights intervals is [-127, 127]) and low precision output precision (on activations is `unsigned int8`, on weigths is `signed int8`),
   - dequantization operations from activations and weights sides were propagated through `Convolution` and were fused in one `Multiply`,
2. `Constant` and `FakeQuantize` operations on weigths were executed and result were stored in `signed int8` result constant.

![](img/fq_and_convolution.transformed.png)


### Quantize and dequantization operations  
In this case `FakeQuantize` operation and `Convert` are used as quantize operation and return quantized low precision tensor. After quantize operation there are `Convert` and dequantization operations to compensate decomposition. Note, please, except `Convert`, all operations are in original precision Original input model:
![](img/qdq_and_convolution.common.png)
LPT result model:  
![](img/qdq_and_convolution.transformed.png)




### Low precision transformations pipeline
LPT transformation pipeline has several steps. You can explore details in [Transformations](#transformations) section below. For each transformation inside one step pattern matcher is unique per transformation, but each operation can be assigned to several transformations.

<details>
<summary>Click to explore all transformations by group in one picture</summary>

![](img/low_precision_transformation_pipeline.png)
</details>

<details>
<summary>Click to explore all transformations by group in one table</summary>

| Step #1: Prerequisites             | Step #2: Markup transformations | Step #3: Main transformations           | Step #4: Cleanup transformations            |
|------------------------------------|---------------------------------|-----------------------------------------|---------------------------------------------|
| PullReshapeThroughDequantization   | MarkupCanBeQuantized            | AddTransformation                       | FoldConvertTransformation                   |
| PullTransposeThroughDequantization | MarkupPrecisions                | AvgPoolTransformation                   | FuseConvertTransformation                   |
| ngraph::pass::LinOpSequenceFusion  | MarkupPerTensorQuantization     | ClampTransformation                     | FuseSubtractToFakeQuantizeTransformation    |
|                                    | MarkupAvgPoolPrecisionPreserved | ConcatTransformation                    | FuseMultiplyToFakeQuantizeTransformation    |
|                                    | PropagatePrecisions             | ConvolutionTransformation               | MultiplyToGroupConvolutionTransformation    |
|                                    | AlignQuantizationInttervals     | ConvolutionBackpropDataTransformation   | SubtractMultiplyToMultiplyAddTransformation |
|                                    | AlignQuantizationParamters      | DepthToSpaceTransformation              | FoldFakeQuantizeTransformation              |
|                                    |                                 | FakeQuantizeDecompositionTransformation |                                             |
|                                    |                                 | FakeQuantizeTransformation              |                                             |
|                                    |                                 | InterpolateTransformation               |                                             |
|                                    |                                 | GroupConvolutionTransformation          |                                             |
|                                    |                                 | MatMulTransformation                    |                                             |
|                                    |                                 | MaxPoolTransformation                   |                                             |
|                                    |                                 | MultiplyTransformation                  |                                             |
|                                    |                                 | MVNTransformation                       |                                             |
|                                    |                                 | NormalizeL2Transformation               |                                             |
|                                    |                                 | PReluTransformation                     |                                             |
|                                    |                                 | ReduceMaxTransformation                 |                                             |
|                                    |                                 | ReduceMeanTransformation                |                                             |
|                                    |                                 | ReduceMinTransformation                 |                                             |
|                                    |                                 | ReduceSumTransformation                 |                                             |
|                                    |                                 | ReluTransformation                      |                                             |
|                                    |                                 | ReshapeTransformation                   |                                             |
|                                    |                                 | SqueezeTransformation                   |                                             |
|                                    |                                 | ShuffleChannelsTransformation           |                                             |
|                                    |                                 | SplitTransformation                     |                                             |
|                                    |                                 | StridedSliceTransformation              |                                             |
|                                    |                                 | TransposeTransformation                 |                                             |
|                                    |                                 | UnsqueezeTransformation                 |                                             |
|                                    |                                 | VariadicSplitTransformation             |                                             |

</details>

Inside each step LPT transformations handle input model operation by operation, applying transformation matching pattern for each transformation from the group to an operation, and execute transformation if pattern is matched. Decomposition transformation decomposes `FakeQuantize` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model is achieved.

As result, usually all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled, and input tensor precisions for the operation will be not changed. In this case the operation is inferred in the original precision. 

Low precision transformations pipeline includes four common steps:
* Step #1: Prerequisites.
* Step #2: Markup transformations.
* Step #3: Main transformations: `FakeQuantize` decomposition and dequantization operations handling.
* Step #4: Cleanup transformations: cleanup result model.

### Step #1. Prerequisites
This step fuses and propagates some operations in the model before run the next step. The step is require for OpenVINO plugins. Transformations:
* PullReshapeThroughDequantization
* PullTransposeThroughDequantization
* ngraph::pass::LinOpSequenceFusion

The model on this step is changed.  

### Step #2. Markup
This step create runtime attributes for operations. Attributes will be used in next step. Transformations:
* MarkupCanBeQuantized
* MarkupPrecisions
* MarkupPerTensorQuantization
* MarkupAvgPoolPrecisionPreserved
* PropagatePrecisions
* AlignQuantizationInttervals
* AlignQuantizationParamters

The model on this step is changed: only new attributes are added to some operations.



### Step #3. Main transformations, FakeQuantize decomposition and dequantization operations handling
This step has the most transformations. Transformations:
* AddTransformation
* AvgPoolTransformation
* ClampTransformation
* ConcatTransformation
* ConvolutionTransformation
* ConvolutionBackpropDataTransformation
* DepthToSpaceTransformation
* FakeQuantizeDecompositionTransformation
* FakeQuantizeTransformation
* InterpolateTransformation
* GroupConvolutionTransformation
* MatMulTransformation
* MaxPoolTransformation
* MultiplyTransformation
* MVNTransformation
* NormalizeL2Transformation
* PReluTransformation
* ReduceMaxTransformation
* ReduceMeanTransformation
* ReduceMinTransformation
* ReduceSumTransformation
* ReluTransformation
* ReshapeTransformation
* SqueezeTransformation
* ShuffleChannelsTransformation
* SplitTransformation
* StridedSliceTransformation
* TransposeTransformation
* UnsqueezeTransformation
* VariadicSplitTransformation

Transformations from this step can be separated in two groups: decompostion transformation and dequantization operations handling.

#### Decomposition transformations
Decomposition transformations decompose `FakeQuantize` operation to quantize (`FakeQuantize` with low precision output) and dequantization operations (revers operations to quantize, with low precision input and original precision output). For dequantization operations LPT uses three operations: `Convert`, `Subtract` and `Multiply`. Element-wise operations `Subtract` and `Multiply` have constants on the second branches. Decomposition transformations have to be executed before other transformations (except branch specific). After decomposition transformations all other transformations in pipeline work with dequantization operations. If dequantization operations are not handled at the end of LPT pipeline, then they will be fused back to the `FakeQuantize`.


Original `FakeQuantize`:  
![TODO: FakeQuantize operation before LPT](quantization/img/fq.common.png)

`FakeQuantize` after decomposition to quantize and dequantization operations:   
![TODO: FakeQuantize operation after LPT](quantization/img/q_and_dq.transformed.png)




#### Dequantization operation handling transformations

In this step LPT transformations fuse or move dequantization operations through existing model operations as more as possible.

Original `Convolution` operation with dequantization operations before:  
![TODO: Convolution operation before LPT](img/qdq_and_convolution.common.png)

`Convolution` operation after decomposition to quantize and dequantization operations:   
![TODO: Convolution operation after LPT](img/qdq_and_convolution.transformed.png)

### Step #4: Cleanup result model
LPT cleanup transformations is final stage in LPT pipeline. In this step LPT transformations clean up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible (fuse at least `Convert` operations if not) to other model operations to cleanup result model. This step is implemented in [cleanup transformations](#cleanup-transformations)

`FakeQuantize` operation with not handled dequantization operations:  
![TODO: FakeQuantize operation with dequantization operations before LPT](img/fq_and_dq.common.png)

`FakeQuantize` operation with fused dequantization operations:  
![TODO: FakeQuantize operation with fused operations after LPT](img/fq_and_dq.transformed.png)

## Low precision transformations in plugin transformation pipeline
Typical transformation pipeline described below.
### Step #1. Common optimizations
This step is optional for LPT but typically is presented in OpenVINO™ plugins. The step doesn't use any LPT transformation. Firstly, the step disables dequantization operations constant folding on constant subgraph on weights to prevent the lost of dequantization info on the next plugin transformations. After that, it optimizes nGraph function and convert operations to operation set 1. Typically, usage of this step is the simplest way to meet LPT requirements for the input quantized model. If plugin can guarantee that LPT input requirements are met, then this step can be skipped.
```cpp
const bool useLpt =
    (conf.lpTransformsMode == Config::LPTransformsMode::On) &&
    ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(nGraphFunc);
if (useLpt) {
    manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
        std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
}

// common transformations
manager.register_pass<ngraph::pass::CommonOptimizations>();

// operation set conversion
manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
...
if (useLpt) {
    manager.register_pass<ngraph::pass::low_precision::ConvertSubtractConstant>(
        std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
}
...
if (useLpt) {
    pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([](const_node_ptr &node) -> bool {
        return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
    });

    pass_config->set_callback<ngraph::pass::ConvertSubtract>([](const_node_ptr &node) -> bool {
        return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
    });
}
manager.run_passes(nGraphFunc);
```
### Step #2. Low precision transformations execution  
This step is mandatory. The step configure and run LPT transformations.
```cpp
if (useLpt) {
    auto supportedPrecisions = std::vector<OperationPrecisionRestriction>({
        OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}},
        }),
        OperationPrecisionRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
            {0, {ngraph::element::u8, ngraph::element::i8}},
            {1, {ngraph::element::i8}}
        }),
        OperationPrecisionRestriction::create<ngraph::opset1::GroupConvolution>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}}
        }),
        OperationPrecisionRestriction::create<ngraph::opset1::Multiply>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}},
        }),
    });

    auto perTensorQuantization = std::vector<OperationPerTensorQuantizationRestriction>({
        OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>({0}),
        OperationPerTensorQuantizationRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0})
    });

    ngraph::pass::Manager lptManager;
    lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);
    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
        if (const auto mulitply = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
        }
        return false;
    });
    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation>([](const_node_ptr& node) -> bool {
        return LayerTransformation::isAsymmetricQuantization(node) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
    });
    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>([](const_node_ptr& node) -> bool {
        return MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node);
    });
    lptManager.run_passes(nGraphFunc);
}
```

### Step #3. Plugin specific transformations  
This step is optional. The step modifies nGraph function to device specific operation set.
```cpp
ngraph::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ngraph::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(nGraphFunc);
```

## Result model overview
Result model depends on different factors:
* The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by POT and NNCF tools. In this case `FakeQuantize` operations are absent before these operations and they will be inferred in original precision.
* LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.

Let explore quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/downloader) tool to download the model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo) :
```sh
./downloader.py --name resnet-50-tf --precisions FP16-INT8
```
If you infer the model in OpenVINO™ CPU plugin, then LPT result model key features are:
* All `FakeQuantize` operations are decomposed and have INT8 output.
* All dequantization operations were handled, moved thought `MaxPool`, `Convolution` and fused with next `FakeQuantize`. As result all input tensor precisions, except one not quantized `SoftMax` operation at the end of the model, were changed to INT8.
> Note, please:
> - LPT transformation for `Add` operation keeps one input branch in FP32. 
> - `Add` operation with one constant branch after `Convolution` is, as expected, still in FP32. It's implementation bias values adding and indivisible part of CPU plugin convolution operation implementation.
> - `FakeQuantize` is quantization operation and has FP32 input as expected.

As result all operations (except not quantized `SoftMax` at the end of the model) in OpenVINO™ CPU plugin are inferred in low precision. Note, please, in the result model there are `FakeQuantize` operations in FP32 but the plugin responsibility is fuse these operations with previous operations. OpenVINO™ CPU plugin achieves maximum optimized inference for all operations by fusing INT8 `Convolution` with FP32 output with `FakeQuantize` operation with FP32 input and INT8 output. In this case OpenVINO™ CPU plugin uses INT8 and FP32 vectorized instructions but reports about one INT8 kernel usage for inference, which is the most optimized for this case.

## Mixed precision
If LPT input model operation output has `fp16` precision then dequantization computations still occurs in `fp32` precision. This approach is used to avoid accuracy loss in `fp16` arithmetic computations. Note, the latest dequantization operation output has `fp16` precision.

## Customization
Transformations can be customizable, each transformation (unless otherwise noted) supports following options:
* Update precisions. Transformation member name is `updatePrecisions`. Boolean value is supported: `true` or `false`. All transformations are affected. If `true` then low precision transformations update precisions to low precision and doesn't if `false`. Typically this option is used for plugin debugging.
* Support asymmetric quantization. Transformation member name is `supportAsymmetricQuantization`. Used in `ConvolutionTransformation` and `GroupConvolution` transformations for weights only. Operation with zero point on weights will be not handled if value is `false`.  
* Precisions on activations. Transformation member name is `precisionsOnActivations`. Array of precisions which define result input precisions for transformed operation.
* Precisions on weights. Transformation member name is `precisionsOnWeights`. Array of precisions which define result input precisions for transformed operation.
* Dequantization precision. Transformation member name is `deqPrecision`.
* Support 3D tensor on activations flag. Transformation member name is ``

## Transformations

LPT transformations are grouped in 4 different groups:
1. branch specific transformations,
2. decomposition transformations,
3. main transformations,
4. cleanup transformations.

It's important to group transformations and use groups in this order. Transformation order inside group doesn't matter.

### Branch specific transformations
There are the following branch specific transformations:
* [ConcatMultiChannelsTransformation](movement/concat_multi_channels.md)
* [ConcatTransformation](movement/concat.md)

### Decomposition transformations
There are the following branch decomposition transformations:
* [FakeQuantizeDecompositionTransformation](quantization/fake_quantize_decomposition.md)

### Main transformations
There are the following main transformations: 
* AddTransformation
* AvgPoolTransformation
* ClampTransformation
* [ConvolutionTransformation](convolution/convolution.md)
* DepthToSpaceTransformation
* FakeQuantizeTransformation
* [GroupConvolutionTransformation](convolution/group_convolution.md)
* InterpolateTransformation
* MatMulTransformation
* MaxPoolTransformation
* MultiplyTransformation
* MVNTransformation
* NormalizeL2Transformation
* PReluTransformation
* ReluTransformation
* ReshapeTransformation
* SqueezeTransformation
* StridedSliceTransformation
* TransposeTransformation
* UnsqueezeTransformation
* InterpolateTransformation

### Clean up transformations
There are the following clean up transformations:
* FoldConvertTransformation
* FuseConvertTransformation
* FuseSubtractToFakeQuantizeTransformation
* FuseMultiplyToFakeQuantizeTransformation
* MultiplyToGroupConvolutionTransformation
* SubtractMultiplyToMultiplyAddTransformation


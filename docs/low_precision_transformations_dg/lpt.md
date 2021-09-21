# OpenVINO™ Low Precision Transformations {#openvino_docs_IE_DG_lpt}

## Attributes

Low precision transformations uses runtime attributes:
* [AvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_AvgPoolPrecisionPreserved)
* [IntervalsAlignment](@ref openvino_docs_IE_DG_lpt_IntervalsAlignment)
* [PerTensorQuantization](@ref openvino_docs_IE_DG_lpt_PerTensorQuantization)
* [PrecisionPreserved](@ref openvino_docs_IE_DG_lpt_PrecisionPreserved)
* [Precisions](@ref openvino_docs_IE_DG_lpt_Precisions)
* [QuantizationAlignment](@ref openvino_docs_IE_DG_lpt_QuantizationAlignment)
* [SharedValueAttribute](@ref openvino_docs_IE_DG_lpt_SharedValueAttribute)

## Transformations

Low precision transformations pipeline includes four steps:
* [Step #1: Prerequisites](@ref openvino_docs_IE_DG_lpt_step1_prerequisites)
* [Step #2: Markup transformations](@ref openvino_docs_IE_DG_lpt_step2_markup)
* [Step #3: Main transformations](@ref openvino_docs_IE_DG_lpt_step3_main)
* [Step #4: Cleanup transformations](@ref openvino_docs_IE_DG_lpt_step4_cleanup)

### Step #1. Prerequisites
This step fuses and propagates some operations in the model before run the next step. The step is required for OpenVINO plugins. Transformations:
* [PullReshapeThroughDequantization](@ref openvino_docs_IE_DG_lpt_PullReshapeThroughDequantization)
* [PullTransposeThroughDequantization](@ref openvino_docs_IE_DG_lpt_PullTransposeThroughDequantization)
* [LinOpSequenceFusion](@ref openvino_docs_IE_DG_lpt_LinOpSequenceFusion)

The model on this step is changed. There are more details in developer guide [Prerequisites transformations](@ref openvino_docs_IE_DG_lpt_step1_prerequisites).

### Step #2. Markup
This step create runtime attributes for operations. Attributes will be used in next step. Transformations:
* [MarkupCanBeQuantized](@ref openvino_docs_IE_DG_lpt_MarkupCanBeQuantized)
* [MarkupPrecisions](@ref openvino_docs_IE_DG_lpt_MarkupPrecisions)
* [MarkupPerTensorQuantization](@ref openvino_docs_IE_DG_lpt_MarkupPerTensorQuantization)
* [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_MarkupAvgPoolPrecisionPreserved)
* [PropagatePrecisions](@ref openvino_docs_IE_DG_lpt_PropagatePrecisions)
* [AlignQuantizationIntervals](@ref openvino_docs_IE_DG_lpt_AlignQuantizationIntervals)
* [AlignQuantizationParameters](@ref openvino_docs_IE_DG_lpt_AlignQuantizationParameters)

The model on this step is changed: only new attributes are added to some operations. There are more details in developer guide [Markup transformations](@ref openvino_docs_IE_DG_lpt_step2_markup).

### Step #3. Main transformations, FakeQuantize decomposition and dequantization operations handling
This step has the most transformations. Transformations:
* [AddTransformation](@ref openvino_docs_IE_DG_lpt_AddTransformation)
* [AvgPoolTransformation](@ref openvino_docs_IE_DG_lpt_AvgPoolTransformation)
* [ClampTransformation](@ref openvino_docs_IE_DG_lpt_AvgPoolTransformation)
* [ConcatTransformation](@ref openvino_docs_IE_DG_lpt_ConcatTransformation)
* [ConvolutionTransformation](@ref openvino_docs_IE_DG_lpt_ConvolutionTransformation)
* [ConvolutionBackpropDataTransformation](@ref openvino_docs_IE_DG_lpt_ConvolutionBackpropDataTransformation)
* [DepthToSpaceTransformation](@ref openvino_docs_IE_DG_lpt_DepthToSpaceTransformation)
* [FakeQuantizeDecompositionTransformation](@ref openvino_docs_IE_DG_lpt_FakeQuantizeDecompositionTransformation)
* [FakeQuantizeTransformation](@ref openvino_docs_IE_DG_lpt_FakeQuantizeTransformation)
* [InterpolateTransformation](@ref openvino_docs_IE_DG_lpt_InterpolateTransformation)
* [GroupConvolutionTransformation](@ref openvino_docs_IE_DG_lpt_GroupConvolutionTransformation)
* [MatMulTransformation](@ref openvino_docs_IE_DG_lpt_MatMulTransformation)
* [MaxPoolTransformation](@ref openvino_docs_IE_DG_lpt_MaxPoolTransformation)
* [MultiplyTransformation](@ref openvino_docs_IE_DG_lpt_MultiplyTransformation)
* [MVNTransformation](@ref openvino_docs_IE_DG_lpt_MVNTransformation)
* [NormalizeL2Transformation](@ref openvino_docs_IE_DG_lpt_NormalizeL2Transformation)
* [PReluTransformation](@ref openvino_docs_IE_DG_lpt_PReluTransformation)
* [ReduceMaxTransformation](@ref openvino_docs_IE_DG_lpt_ReduceMaxTransformation)
* [ReduceMeanTransformation](@ref openvino_docs_IE_DG_lpt_ReduceMeanTransformation)
* [ReduceMinTransformation](@ref openvino_docs_IE_DG_lpt_ReduceMinTransformation)
* [ReduceSumTransformation](@ref openvino_docs_IE_DG_lpt_ReduceSumTransformation)
* [ReluTransformation](@ref openvino_docs_IE_DG_lpt_ReluTransformation)
* [ReshapeTransformation](@ref openvino_docs_IE_DG_lpt_ReshapeTransformation)
* [SqueezeTransformation](@ref openvino_docs_IE_DG_lpt_SqueezeTransformation)
* [ShuffleChannelsTransformation](@ref openvino_docs_IE_DG_lpt_ShuffleChannelsTransformation)
* [SplitTransformation](@ref openvino_docs_IE_DG_lpt_SplitTransformation)
* [StridedSliceTransformation](@ref openvino_docs_IE_DG_lpt_StridedSliceTransformation)
* [TransposeTransformation](@ref openvino_docs_IE_DG_lpt_TransposeTransformation)
* [UnsqueezeTransformation](@ref openvino_docs_IE_DG_lpt_UnsqueezeTransformation)
* [VariadicSplitTransformation](@ref openvino_docs_IE_DG_lpt_VariadicSplitTransformation)

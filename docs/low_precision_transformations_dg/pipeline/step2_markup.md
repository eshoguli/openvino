# OpenVINO™ LPT: step #2. Markup transformations {#openvino_docs_IE_DG_lpt_step2_markup}

This step defines the most optimal `FakeQuantize` decomposition precisions for the best inference performance via operations markup with runtime attribute instances. Attributes are created for input & output ports and operations. Operation output port precisions are not changed in these transformations. Transformations order is important. A model markup low precision logic is decomposed and implemented into following common markup transformations (in usage order, order is important):
1. [MarkupCanBeQuantized](@ref openvino_docs_IE_DG_lpt_MarkupCanBeQuantized)
2. [MarkupPrecisions](@ref openvino_docs_IE_DG_lpt_MarkupPrecisions)
3. [MarkupPerTensorQuantization](@ref openvino_docs_IE_DG_lpt_MarkupPerTensorQuantization)
4. [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_MarkupAvgPoolPrecisionPreserved)
5. [PropagatePrecisions](@ref openvino_docs_IE_DG_lpt_PropagatePrecisions)
6. [AlignQuantizationIntervals](@ref openvino_docs_IE_DG_lpt_AlignQuantizationIntervals)
7. [AlignQuantizationParameters](@ref openvino_docs_IE_DG_lpt_AlignQuantizationParameters)

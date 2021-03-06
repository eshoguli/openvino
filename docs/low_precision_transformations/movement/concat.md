# OpenVINO™ Low Precision Transformations

## Table of Contents
1. [Introduction](#introduction)
2. [Intermediate operations before Concat](#intermediate-operations-before-concat)
3. [Intermediate operations before Concat with output](#intermediate-operations-before-concat-with-output)
4. [Several Concat operations with one common FakeQuantize (not cascading)](#several-concat-operations-with-one-common-fakequantize-not-cascading)
5. [Cascade Concat operations with intermediate operations](#cascade-concat-operations-with-intermediate-operations)

## Introduction
`ConcatTransformation` transformation decompose `FakeQuantize` operations and move dequantization operations through `Concat` operation. As result `Concat` operations are inferred in low precision and propagate low precision to the next operations. The difference from `ConcatMultiChannelsTransformation` is per-tensor quantization support only.

## Intermediate operations before Concat
`MaxPool` and `Interpolate` are intermediate operations before `Concat`. 
Before transformations:

![](img/fq_and_max_pool_and_concat.common.png)

After transformations:

![](img/fq_and_max_pool_and_concat.transformed.png)

## Intermediate operations before Concat with output
`MaxPool` is intermediate operation before `Concat`. 

Before transformations:

![](img/fq_and_max_pool_with_output_and_concat.common.png)

After transformations:

![](img/fq_and_max_pool_with_output_and_concat.transformed.png)

## Several Concat operations with one common FakeQuantize (not cascading)
Several `Concat` operations have on common `FakeQuantize`. Note, please, `Concat` operations are not cascading. 

Before transformations:

![](img/fq_and_two_concat_with_one_parent.common.png)

After transformations:

![](img/fq_and_two_concat_with_one_parent.transformed.png)

## Cascade Concat operations with intermediate operations
Before transformations:

![](img/fq_and_two_cascade_concats_with_intermediate.common.png)

After transformations:

![](img/fq_and_two_cascade_concats_with_intermediate.transformed.png)
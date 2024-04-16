### Context

CPU plugin just-in-time (JIT) emitters implement element-wise operations. Use [CPU plugin JIT emitters](https://github.com/eshoguli/openvino/blob/es/cpu/emitters_doc/src/plugins/intel_cpu/src/emitters/README.md) documentation for details.

### What needs to be done?

* Create [Mod](https://docs.openvino.ai/2023.3/openvino_docs_ops_arithmetic_Mod_1.html) operation `float` type JIT emitter for ARM64 SIMD platform. Use OpenVINO [CPU plugin JIT emitters](https://github.com/eshoguli/openvino/blob/es/cpu/emitters_doc/src/plugins/intel_cpu/src/emitters/README.md) documentation and [Arm A-profile A64 Instruction Set Architecture](https://developer.arm.com/documentation/ddi0602/2024-03/SIMD-FP-Instructions) documentation for details.
* Modify [ARM64 executor](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/src/nodes/executors/aarch64/jit_eltwise.cpp#L20) and [ARM64 kernel](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/src/nodes/kernels/aarch64/jit_uni_eltwise_generic.hpp#L80) (in both places: [create_eltwise_emitter](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/src/nodes/kernels/aarch64/jit_uni_eltwise_generic.cpp#L601) and [get_supported_precisions](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/src/nodes/kernels/aarch64/jit_uni_eltwise_generic.cpp#L766)) to apply developed JIT emitter.

Before emitter implementation, please, modify tests to be sure that deveoped functionality is covered by test:
  * Add `jit` kernel check in [element-wise](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/classes/eltwise.cpp#L257) or [activation](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/classes/activation.cpp#L155) test, the test must fail.
  * If test doesn't fail then add supported operation in [element-wise test instantiation](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/classes/eltwise.cpp) or [activation test instantiation](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/classes/activation.cpp) if it's absent.

### Tests
[GoogleTest](https://github.com/google/googletest) is used for testing. CPU functional test target is [ov_cpu_func_tests](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/tests/functional). You can use two `GoogleTest` filters for element-wise and activation operations:
* for element-wise operations:
```sh
./bin/[platform]/[build_type]/ov_cpu_func_tests --gtest_filter="*smoke*Eltwise*"
```
* for activations:
```sh
./bin/[platform]/[build_type]/ov_cpu_func_tests --gtest_filter="*smoke*Activation*"
```

### Example Pull Requests

* [Abs](https://docs.openvino.ai/2023.3/openvino_docs_ops_arithmetic_Abs_1.html) operation: https://github.com/openvinotoolkit/openvino/pull/23692
* [Sigmoid](https://docs.openvino.ai/2023.3/openvino_docs_ops_activation_Sigmoid_1.html) operation, which reuses [Exp emitter](https://github.com/openvinotoolkit/openvino/blob/4ef40f0f0e70abf04f13310f3e7a8f6784667dbf/src/plugins/intel_cpu/src/emitters/plugin/aarch64/jit_eltwise_emitters.cpp#L820): https://github.com/openvinotoolkit/openvino/pull/23065

### Resources

- [Contribution guide - start here!](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)
- [What is OpenVINO?](https://github.com/openvinotoolkit/openvino#what-is-openvino-toolkit)
- [CPU plugin JIT emitters](https://github.com/eshoguli/openvino/blob/es/cpu/emitters_doc/src/plugins/intel_cpu/src/emitters/README.md)
- [Blog post on contributing to OpenVINO](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)
- [User documentation](https://docs.openvino.ai/)

### Contact points

@eshoguli, @dmitry-gorokhov

### Ticket

CVS-137689

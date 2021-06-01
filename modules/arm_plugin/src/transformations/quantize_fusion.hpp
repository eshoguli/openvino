// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

namespace ArmPlugin {
namespace pass {

class FakeQuantizeFusionBase: public ngraph::pass::MatcherPass {
protected:
    template <class Node>
    void registerMatcher(const std::string& name);
};

class ConvFakeQuantizeFusion: public FakeQuantizeFusionBase {
public:
    ConvFakeQuantizeFusion();
};

class GroupConvFakeQuantizeFusion: public FakeQuantizeFusionBase {
public:
    GroupConvFakeQuantizeFusion();
};

class MatMulFakeQuantizeFusion: public FakeQuantizeFusionBase {
public:
    MatMulFakeQuantizeFusion();
};

class AvgPoolFakeQuantizeFusion: public FakeQuantizeFusionBase {
public:
    AvgPoolFakeQuantizeFusion();
};

class QuantizeFusion: public ngraph::pass::GraphRewrite {
public:
    QuantizeFusion() {
        add_matcher<AvgPoolFakeQuantizeFusion>();
        add_matcher<ConvFakeQuantizeFusion>();
        add_matcher<GroupConvFakeQuantizeFusion>();
        add_matcher<MatMulFakeQuantizeFusion>();
    }
};

class PropogateQuantizationInfo: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

class FixQuantizedSubtractOutputDataType: public ngraph::pass::MatcherPass {
public:
    FixQuantizedSubtractOutputDataType();
};

class FixMulInputConvert: public ngraph::pass::MatcherPass {
public:
    FixMulInputConvert();
};

class DetectMaybeQuantized: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

}  // namespace pass
}  // namespace ArmPlugin

namespace ngraph {
template <>
struct NGRAPH_API VariantWrapper<ngraph::element::Type> : public VariantImpl<ngraph::element::Type> {
    NGRAPH_RTTI_DECLARATION;
    VariantWrapper(const ngraph::element::Type& value) : VariantImpl<ngraph::element::Type>{value} {}
    ~VariantWrapper() override;
};
}  // namespace ngraph

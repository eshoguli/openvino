// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_low_precision_transformer.hpp"

#include <string>
#include <ngraph/ngraph.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/transformation_context.hpp>
#include <low_precision/layer_transformation.hpp>
#include <low_precision/transformation_context.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/align_quantization_parameters.hpp>

using namespace testing;
using namespace ngraph::pass;

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer(const std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>& precisions) {
    auto passConfig = std::make_shared<PassConfig>();
    lowPrecisionManager = std::make_shared<ngraph::pass::Manager>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupPrecisions>(precisions);
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();

//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisions);
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming1.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming1").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming2.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming2").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming3.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming4.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming4").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming5.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming5").run_on_function(f);
//    }

    this->common = lowPrecisionManager->register_pass<ngraph::pass::GraphRewrite>();
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ngraph::Function>& function) {
    ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer pass;
    pass.run_on_function(function);

    context.function = function;
    lowPrecisionManager->run_passes(function);
}

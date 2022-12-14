// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/update_precision.hpp"

#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "ngraph/op/util/multi_subgraph_base.hpp"

#include <transformations/utils/utils.hpp>
#include <low_precision/lpt_itt.hpp>
#include <low_precision/network_helper.hpp>

#define SET_OUT_DATA_PRECISION(TYPE_NAME) \
    if (ngraph::as_type_ptr<TYPE_NAME>(node)) { \
        auto op = std::dynamic_pointer_cast<TYPE_NAME>(node); \
        NetworkHelper::setOutDataPrecision(op, keep_precision); \
    } \

ngraph::pass::low_precision::UpdatePrecision::UpdatePrecision(const ngraph::element::Type keep_precision) : 
    keep_precision(keep_precision) {}

bool ngraph::pass::low_precision::UpdatePrecision::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "UpdatePrecision");

    bool was_changed = false;
    for (const auto& node : f->get_ordered_ops()) {
        if (transformation_callback(node) || 
            ov::as_type_ptr<ngraph::opset1::Parameter>(node) ||
            ov::as_type_ptr<ngraph::opset1::Result>(node)) {
            continue;
        }

        bool one_result = false;
        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                if (ov::is_type<ngraph::opset1::Result>(input.get_node())) {
                    one_result = true;
                    break;
                }
            }
            if (one_result) {
                break;
            }
        }

        if (one_result) {
            continue;
        }

        for (const auto& output : node->outputs()) {
            if (output.get_element_type() != ov::element::f32) {
                continue;
            }

            auto constant = ov::as_type_ptr<ngraph::opset1::Constant>(node);
            if (constant != nullptr) {
                auto new_constant = NetworkHelper::foldDequantizationConstant(
                    constant,
                    std::make_shared<ngraph::opset1::Convert>(constant, keep_precision),
                    0);
                ngraph::replace_node(constant, {new_constant->output(0)});
                continue;
            }

            auto convert = ov::as_type_ptr<ngraph::opset1::Convert>(node);
            if (convert != nullptr) {
                auto new_convert = std::make_shared<ngraph::opset1::Convert>(
                    convert->get_input_source_output(0), 
                    keep_precision);
                ngraph::replace_node(convert, {new_convert->output(0)});
                continue;
            }
            
            // all operations are from main & clean up transformations list
            SET_OUT_DATA_PRECISION(ngraph::opset1::Add)
            SET_OUT_DATA_PRECISION(ngraph::opset3::Assign)
            SET_OUT_DATA_PRECISION(ngraph::opset6::Assign)
            SET_OUT_DATA_PRECISION(ngraph::opset1::AvgPool)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Clamp)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Concat)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Convolution)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ConvolutionBackpropData)
            SET_OUT_DATA_PRECISION(ngraph::opset1::DepthToSpace)
            SET_OUT_DATA_PRECISION(ngraph::opset1::FakeQuantize)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Interpolate)
            SET_OUT_DATA_PRECISION(ngraph::opset1::GroupConvolution)
            SET_OUT_DATA_PRECISION(ngraph::opset1::MatMul)
            SET_OUT_DATA_PRECISION(ngraph::opset1::MaxPool)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Multiply)
            SET_OUT_DATA_PRECISION(op::MVN)
            SET_OUT_DATA_PRECISION(ngraph::opset6::MVN)
            SET_OUT_DATA_PRECISION(ngraph::opset1::NormalizeL2)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Pad)
            SET_OUT_DATA_PRECISION(ngraph::opset1::PRelu)
            SET_OUT_DATA_PRECISION(ngraph::opset1::LSTMSequence)
            SET_OUT_DATA_PRECISION(ngraph::opset5::GRUSequence)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ReduceMax)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ReduceMean)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ReduceMin)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ReduceSum)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Relu)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Reshape)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Squeeze)
            SET_OUT_DATA_PRECISION(ngraph::opset1::ShuffleChannels)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Split)
            SET_OUT_DATA_PRECISION(ngraph::opset1::StridedSlice)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Transpose)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Unsqueeze)
            SET_OUT_DATA_PRECISION(ngraph::opset1::VariadicSplit)
            SET_OUT_DATA_PRECISION(ngraph::opset4::Interpolate)

            SET_OUT_DATA_PRECISION(ngraph::opset1::Convert)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Subtract)
            SET_OUT_DATA_PRECISION(ngraph::opset1::Multiply)
            break;
        }

        was_changed = true;
    }

    return was_changed;
}

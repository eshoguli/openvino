// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/transformation_context.hpp"
#include "low_precision/transformer.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8U8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();

    static std::string toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type& type,
        const ngraph::Shape& shape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static ngraph::builder::subgraph::DequantizationOperations toDequantizationOperations(
        const ngraph::pass::low_precision::FakeQuantizeDequantization& dequantization);

    template <class Operation>
    static std::vector<std::shared_ptr<ngraph::Node>> get(std::shared_ptr<ngraph::Function> function) {
        std::vector<std::shared_ptr<ngraph::Node>> foundNodes;
        std::vector<std::shared_ptr<ngraph::Node>>& nodes = function->get_ordered_ops();
        for (auto& node : nodes) {
            if (is_type<Operation>(node)) {
                foundNodes.push_back(node);
            }
        }
        return foundNodes;
    }

    template <class Attribute>
    static bool checkIfOutputAttributesAreEqual(std::vector<std::shared_ptr<ngraph::Node>> nodes) {
        Variant* first = nullptr;
        for (auto node : nodes) {
            for (auto output : node->outputs()) {
                auto& rt = output.get_rt_info();
                const std::string& name = ngraph::VariantWrapper<Attribute>::type_info.name;
                auto it = rt.find(name);
                if (it == rt.end()) {
                    return false;
                }

                auto value = it->second;
                if (first == nullptr) {
                    first = value.get();
                } else if(value.get() != first) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class Attribute>
    static bool checkIfAttributesAreEqual(std::vector<std::shared_ptr<ngraph::Node>> nodes) {
        Variant* first = nullptr;
        for (auto node : nodes) {
            auto& rt = node->get_rt_info();
            const std::string& name = ngraph::VariantWrapper<Attribute>::type_info.name;
            auto it = rt.find(name);
            if (it == rt.end()) {
                return false;
            }

            auto value = it->second;
            if (first == nullptr) {
                first = value.get();
            }
            else if (value.get() != first) {
                return false;
            }
        }
        return true;
    }

protected:
    void transform(std::shared_ptr<ngraph::Function> function);
    void transform(
        std::shared_ptr<ngraph::Function> function,
        std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr>& transformations);

    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "base_matcher_pass.hpp"
#include "network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

template <typename AttributeType, typename OperationType>
class CreateAttribute;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Attribute::init(Node&) <= can we reuse?
enum class AttributeSource {
    Node,
    OutputPort
};

template <typename AttributeType, typename OperationType = ngraph::pattern::op::Label>
class ngraph::pass::low_precision::CreateAttribute : public ngraph::pass::low_precision::BaseMatcherPass {
public:
    CreateAttribute(const AttributeSource source = AttributeSource::Node) {
        assert((source == AttributeSource::Node) || (source == AttributeSource::OutputPort));
        auto operation = std::is_same<OperationType, pattern::op::Label>::value ?
            std::make_shared<pattern::op::Label>(element::f32, Shape{}, [](std::shared_ptr<Node> n) { return true; }) :
            pattern::wrap_type<OperationType>();

        ngraph::graph_rewrite_callback callback = [&](pattern::Matcher& m) {
            auto op = m.get_match_root();
            if (!op || transformation_callback(op)) {
                return false;
            }
            auto attribute = ngraph::VariantWrapper<std::shared_ptr<AttributeType>>::create(op, params);
            if (attribute == nullptr) {
                return false;
            }

            return true;
        };

        auto matcher = std::make_shared<ngraph::pattern::Matcher>(operation, "CreateAttribute");
        this->register_matcher(matcher, callback);
    }
};

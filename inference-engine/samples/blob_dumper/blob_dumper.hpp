// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief message for help argument
static const char help_message[] = "Print a usage message";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml/.onnx/.prototxt file with a trained model or to a .blob files with a trained compiled model.";

/// @brief message for dump directory argument
static const char dump_dir_message[] = "Required. Directory to store dumps.";

/// @brief message for reduce argument
static const char reduce_message[] = "Optional. Reduce blobs.";

/// @brief message for rename_message argument
static const char rename_message[] = "Optional. Rename operation names.";

/// @brief message for ignore_const_message argument
static const char ignore_const_message[] = "Optional. Ignore Const operations.";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Declare flag for showing help message <br>
DECLARE_bool(help);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(model, "", model_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(dump_dir, "", dump_dir_message);

/// @brief Define parameter to reduce blobs <br>
/// It is a optional parameter
DEFINE_bool(reduce, false, reduce_message);

/// @brief Define parameter to rename operation names <br>
/// It is a optional parameter
DEFINE_bool(rename, false, rename_message);

/// @brief Define parameter to ignore constant operation dump<br>
/// It is a optional parameter
DEFINE_bool(ignore_const, false, ignore_const_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "benchmark_app [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h, --help                " << help_message << std::endl;
    std::cout << "    -model \"<path>\"         " << model_message << std::endl;
    std::cout << "    -dump_dir \"<directory>\" " << dump_dir_message << std::endl;
    std::cout << "    -reduce                   " << reduce_message << std::endl;
}

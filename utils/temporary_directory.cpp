// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "temporary_directory.h"
#include "utils/gputils.h"
#include <cstdlib>

namespace Utils
{
TemporaryDirectory::TemporaryDirectory(const std::filesystem::path &parent)
{
    auto pattern = (parent / ".mtp-workflow-XXXXXX").string();
    if (!mkdtemp(pattern.data()))
        GPERROR("Cannot stage MTP workflow output");
    path = pattern;
}

TemporaryDirectory::~TemporaryDirectory()
{
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}
}

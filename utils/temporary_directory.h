// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <filesystem>

namespace Utils
{
class TemporaryDirectory
{
  public:
    std::filesystem::path path;
    explicit TemporaryDirectory(
        const std::filesystem::path &parent = std::filesystem::temp_directory_path());
    ~TemporaryDirectory();
    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
};
}

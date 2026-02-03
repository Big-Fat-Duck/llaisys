#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {

void embedding(std::byte* out,
               const std::byte* index,
               const std::byte* weight,
               llaisysDataType_t type,
               size_t n,
               size_t vocab,
               size_t dim);
} // namespace llaisys::ops::cpu
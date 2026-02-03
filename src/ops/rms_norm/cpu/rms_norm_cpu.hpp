#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {

void rms_norm(std::byte* out,
              const std::byte* in,
              const std::byte* weight,
              llaisysDataType_t type,
              size_t M,
              size_t D,
              float eps);
} // namespace llaisys::ops::cpu
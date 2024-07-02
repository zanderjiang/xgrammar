/*!
 * Copyright (c) 2023 by Contributors
 * \file support/utils.h
 * \brief Utility functions.
 */
#ifndef XGRAMMAR_SUPPORT_UTILS_H_
#define XGRAMMAR_SUPPORT_UTILS_H_

#include <cstdint>

namespace xgrammar {

/*!
 * \brief Hash and combine value into seed.
 * \ref https://www.boost.org/doc/libs/1_84_0/boost/intrusive/detail/hash_combine.hpp
 */
inline void HashCombineBinary(uint32_t& seed, uint32_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/*!
 * \brief Find the hash sum of several uint32_t args.
 */
template <typename... Args>
uint32_t HashCombine(Args... args) {
  uint32_t seed = 0;
  (..., HashCombineBinary(seed, args));
  return seed;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UTILS_H_

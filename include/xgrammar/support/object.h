/*!
 *  Copyright (c) 2023 by Contributors
 * \file support/encoding.h
 * \brief Encoding and decoding from/to UTF-8 and escape sequence to/from codepoints.
 */
#ifndef XGRAMMAR_SUPPORT_OBJECT_H_
#define XGRAMMAR_SUPPORT_OBJECT_H_

namespace xgrammar {

#define XGRAMMAR_DEFINE_PIMPL_METHODS(TypeName)                            \
 public:                                                                   \
  class Impl;                                                              \
  TypeName() = default;                                                    \
  explicit TypeName(const std::shared_ptr<Impl>& pimpl) : pimpl_(pimpl) {} \
  TypeName(const TypeName& other) = default;                               \
  TypeName(TypeName&& other) noexcept = default;                           \
  TypeName& operator=(const TypeName& other) = default;                    \
  TypeName& operator=(TypeName&& other) noexcept = default;                \
  Impl* operator->() { return pimpl_.get(); }                              \
                                                                           \
 private:                                                                  \
  std::shared_ptr<Impl> pimpl_

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_OBJECT_H_

#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <stdexcept>
#include <complex>
#include <string>

/**
 * @brief Template struct for mapping C++ types to ACL data types
 * 
 * Provides compile-time conversion between C++ primitive types and ACLDataType enum values.
 * Specializations ensure proper mapping for supported data types.
 */
template <typename T>
struct TypeToACLDtype;

// Specializations for supported types
template <> struct TypeToACLDtype<float> { static constexpr aclDataType value = ACL_FLOAT; };
template <> struct TypeToACLDtype<double> { static constexpr aclDataType value = ACL_DOUBLE; };
template <> struct TypeToACLDtype<int8_t> { static constexpr aclDataType value = ACL_INT8; };
template <> struct TypeToACLDtype<int16_t> { static constexpr aclDataType value = ACL_INT16; };
template <> struct TypeToACLDtype<int32_t> { static constexpr aclDataType value = ACL_INT32; };
template <> struct TypeToACLDtype<int64_t> { static constexpr aclDataType value = ACL_INT64; };
template <> struct TypeToACLDtype<uint8_t> { static constexpr aclDataType value = ACL_UINT8; };
template <> struct TypeToACLDtype<uint16_t> { static constexpr aclDataType value = ACL_UINT16; };
template <> struct TypeToACLDtype<uint32_t> { static constexpr aclDataType value = ACL_UINT32; };
template <> struct TypeToACLDtype<uint64_t> { static constexpr aclDataType value = ACL_UINT64; };
template <> struct TypeToACLDtype<bool> { static constexpr aclDataType value = ACL_BOOL; };

/**
 * @brief Creates a scalar object with automatic type deduction
 * @tparam T Type of the value to store in the scalar (automatically deduced)
 * @param value Scalar value to store
 * @return aclScalar* Pointer to the created scalar object
 */
template <typename T>
aclScalar* CreateScalar(T value);

/**
 * @brief Creates a scalar object with explicit data type specification
 * @tparam ValueType Type of the input value (automatically deduced)
 * @param value Scalar value to store
 * @param dtype Target ACL data type for the scalar
 * @return aclScalar* Pointer to the created scalar object
 * @throws std::runtime_error If the specified data type is unsupported
 * @note For best performance, match ValueType with the ACL data type representation:
 *       - ACL_FLOAT: float
 *       - ACL_DOUBLE: double
 *       - ACL_INTx: matching intx_t type
 *       - ACL_UINTx: matching uintx_t type
 *       - ACL_BOOL: bool
 */
template <typename ValueType>
aclScalar* CreateScalar(ValueType value, aclDataType dtype);


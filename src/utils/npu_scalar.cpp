#include "asnumpy/utils/npu_scalar.hpp"

/*
    Creates an aclScalar object by automatically determining the appropriate ACL data type
    based on the C++ type of the input value. Uses TypeToACLDtype for compile-time mapping.
*/
template <typename T>
aclScalar* CreateScalar(T value) {
    return CreateScalar(value, TypeToACLDtype<std::decay_t<T>>::value);
}


/*
    Creates an aclScalar object for a given value with explicit data type control.
    Performs optimized value conversion when the input type matches the target data type.
    Falls back to static_cast conversion when types differ.
*/
template <typename ValueType>
aclScalar* CreateScalar(ValueType value, aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, float>) {
                return aclCreateScalar(&value, ACL_FLOAT);
            } else {
                auto converted = static_cast<float>(value);
                return aclCreateScalar(&converted, ACL_FLOAT);
            }
        }
        case ACL_DOUBLE: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, double>) {
                return aclCreateScalar(&value, ACL_DOUBLE);
            } else {
                auto converted = static_cast<double>(value);
                return aclCreateScalar(&converted, ACL_DOUBLE);
            }
        }
        case ACL_INT32: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int32_t>) {
                return aclCreateScalar(&value, ACL_INT32);
            } else {
                auto converted = static_cast<int32_t>(value);
                return aclCreateScalar(&converted, ACL_INT32);
            }
        }
        case ACL_INT64: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int64_t>) {
                return aclCreateScalar(&value, ACL_INT64);
            } else {
                auto converted = static_cast<int64_t>(value);
                return aclCreateScalar(&converted, ACL_INT64);
            }
        }
        case ACL_INT8: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int8_t>) {
                return aclCreateScalar(&value, ACL_INT8);
            } else {
                auto converted = static_cast<int8_t>(value);
                return aclCreateScalar(&converted, ACL_INT8);
            }
        }
        case ACL_INT16: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int16_t>) {
                return aclCreateScalar(&value, ACL_INT16);
            } else {
                auto converted = static_cast<int16_t>(value);
                return aclCreateScalar(&converted, ACL_INT16);
            }
        }
        case ACL_UINT8: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint8_t>) {
                return aclCreateScalar(&value, ACL_UINT8);
            } else {
                auto converted = static_cast<uint8_t>(value);
                return aclCreateScalar(&converted, ACL_UINT8);
            }
        }
        case ACL_UINT16: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint16_t>) {
                return aclCreateScalar(&value, ACL_UINT16);
            } else {
                auto converted = static_cast<uint16_t>(value);
                return aclCreateScalar(&converted, ACL_UINT16);
            }
        }
        case ACL_UINT32: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint32_t>) {
                return aclCreateScalar(&value, ACL_UINT32);
            } else {
                auto converted = static_cast<uint32_t>(value);
                return aclCreateScalar(&converted, ACL_UINT32);
            }
        }
        case ACL_UINT64: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint64_t>) {
                return aclCreateScalar(&value, ACL_UINT64);
            } else {
                auto converted = static_cast<uint64_t>(value);
                return aclCreateScalar(&converted, ACL_UINT64);
            }
        }
        case ACL_BOOL: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, bool>) {
                return aclCreateScalar(&value, ACL_BOOL);
            } else {
                auto converted = static_cast<bool>(value);
                return aclCreateScalar(&converted, ACL_BOOL);
            }
        }
        case ACL_FLOAT16: {
            auto converted = static_cast<uint16_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT16);
        }
        case ACL_BF16: {
            auto converted = static_cast<uint16_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_BF16);
        }
        case ACL_INT4: {
            auto converted = static_cast<int8_t>(value);
            return aclCreateScalar(&converted, ACL_INT4);
        }
        case ACL_UINT1: {
            auto converted = static_cast<uint8_t>(value);
            return aclCreateScalar(&converted, ACL_UINT1);
        }
        case ACL_COMPLEX64: {
            auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
            return aclCreateScalar(&converted, ACL_COMPLEX64);
        }
        case ACL_COMPLEX128: {
            auto converted = std::complex<double>(static_cast<double>(value), 0.0);
            return aclCreateScalar(&converted, ACL_COMPLEX128);
        }
        case ACL_COMPLEX32: {
            auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
            return aclCreateScalar(&converted, ACL_COMPLEX32);
        }
        case ACL_HIFLOAT8: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_HIFLOAT8);
        }
        case ACL_FLOAT8_E5M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E5M2);
        }
        case ACL_FLOAT8_E4M3FN: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E4M3FN);
        }
        case ACL_FLOAT8_E8M0: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E8M0);
        }
        case ACL_FLOAT6_E3M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT6_E3M2);
        }
        case ACL_FLOAT6_E2M3: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT6_E2M3);
        }
        case ACL_FLOAT4_E2M1: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT4_E2M1);
        }
        case ACL_FLOAT4_E1M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT4_E1M2);
        }
        case ACL_STRING: {
            // 对于字符串类型，创建一个简单的字符串表示
            std::string str_value = std::to_string(static_cast<double>(value));
            // 使用const_cast来移除const限定符，因为aclCreateScalar需要void*
            return aclCreateScalar(const_cast<char*>(str_value.c_str()), ACL_STRING);
        }
        case ACL_DT_UNDEFINED: {
            // 对于未定义类型，使用默认值0
            auto converted = static_cast<int32_t>(0);
            return aclCreateScalar(&converted, ACL_INT32);
        }
        default:
            throw std::runtime_error("Unsupported dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}
#pragma once
// Tensor metadata + cl_mem wrapper used by the ONNX→OpenCL runner.

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace ocl {

// ONNX TensorProto.DataType subset we care about.
enum class DType : int {
    FLOAT32 = 1,
    UINT8   = 2,
    INT8    = 3,
    UINT16  = 4,
    INT16   = 5,
    INT32   = 6,
    INT64   = 7,
    BOOL    = 9,
    FLOAT16 = 10,
    UNKNOWN = 0,
};

inline DType dtype_from_str(const std::string& s) {
    if (s == "float32") return DType::FLOAT32;
    if (s == "uint8")   return DType::UINT8;
    if (s == "int8")    return DType::INT8;
    if (s == "uint16")  return DType::UINT16;
    if (s == "int16")   return DType::INT16;
    if (s == "int32")   return DType::INT32;
    if (s == "int64")   return DType::INT64;
    if (s == "bool")    return DType::BOOL;
    if (s == "float16") return DType::FLOAT16;
    return DType::UNKNOWN;
}

inline std::string dtype_to_str(DType d) {
    switch (d) {
        case DType::FLOAT32: return "float32";
        case DType::UINT8:   return "uint8";
        case DType::INT8:    return "int8";
        case DType::UINT16:  return "uint16";
        case DType::INT16:   return "int16";
        case DType::INT32:   return "int32";
        case DType::INT64:   return "int64";
        case DType::BOOL:    return "bool";
        case DType::FLOAT16: return "float16";
        default:             return "unknown";
    }
}

inline int dtype_size(DType d) {
    switch (d) {
        case DType::FLOAT32: case DType::INT32:                       return 4;
        case DType::UINT8:   case DType::INT8:   case DType::BOOL:    return 1;
        case DType::UINT16:  case DType::INT16:  case DType::FLOAT16: return 2;
        case DType::INT64:                                            return 8;
        default:                                                      return 0;
    }
}

struct Tensor {
    std::string name;
    DType dtype = DType::FLOAT32;
    std::vector<int64_t> shape;        // may include 0 for symbolic / unknown
    cl_mem device = nullptr;           // owned; null if not allocated yet
    size_t byte_size = 0;
    // Optional fp32 host shadow (used only for tiny tensors like Reshape's shape input).
    std::vector<uint8_t> host_bytes;

    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (auto d : shape) {
            if (d <= 0) return 0;
            n *= d;
        }
        return n;
    }

    std::string shape_str() const {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            os << shape[i];
            if (i + 1 < shape.size()) os << ",";
        }
        os << "]";
        return os.str();
    }
};

} // namespace ocl

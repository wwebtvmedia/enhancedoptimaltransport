#pragma once
// OpenCL runtime that walks an ONNX graph (loaded via Manifest) and dispatches
// each node to an op handler. Per-op logging + optional golden-diff.

#include "manifest.hpp"
#include "tensor.hpp"
#include "logger.hpp"

#include <CL/cl.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ocl {

struct DiffStats {
    bool found_golden = false;
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    double max_rel_err = 0.0;
    int64_t argmax = 0;
    double golden_at_argmax = 0.0;
    double got_at_argmax = 0.0;
};

class Runtime {
public:
    Logger log;

    Runtime() = default;
    ~Runtime() { cleanup(); }

    // --- Setup ---
    void init_opencl();
    void load_manifest(const std::string& manifest_path);
    void load_initializers();   // Reads init/*.bin into device buffers.
    void compile_kernels(const std::string& cl_source_path);

    // --- Inputs ---
    void set_input_fp32(const std::string& name,
                        const std::vector<int64_t>& shape,
                        const float* data);

    // --- Run ---
    // Walk all nodes; if golden_dir is non-empty, diff each output against
    // golden_dir/<safe_name>.bin and log result.
    // Stops at first failing op unless continue_on_error=true.
    void run(const std::string& golden_dir = "", double abs_tol = 1e-4,
             double rel_tol = 1e-3, bool continue_on_error = true);

    // --- Outputs ---
    void read_output_fp32(const std::string& name, std::vector<float>& out);

    void cleanup();

    // Accessors
    const Manifest& manifest() const { return manifest_; }

private:
    // --- OpenCL state ---
    cl_platform_id platform_ = nullptr;
    cl_device_id   device_   = nullptr;
    cl_context     context_  = nullptr;
    cl_command_queue queue_  = nullptr;
    cl_program     program_  = nullptr;
    std::unordered_map<std::string, cl_kernel> kernels_;

    // --- Model state ---
    Manifest manifest_;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> tensors_;

    // Per-op stats (running tally).
    int ops_done_ = 0;
    int ops_failed_ = 0;
    int ops_skipped_ = 0;

    // --- Helpers ---
    Tensor* get_tensor(const std::string& name);
    Tensor* require_tensor(const std::string& name);
    Tensor* alloc_tensor(const std::string& name, DType dtype,
                         const std::vector<int64_t>& shape);
    cl_mem  alloc_buffer(size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void    upload(cl_mem buf, const void* data, size_t bytes);
    void    download(cl_mem buf, void* data, size_t bytes);

    cl_kernel kernel(const std::string& name);

    void run_node(const NodeInfo& n);

    // --- Op handlers ---
    void op_dequantize_linear(const NodeInfo& n);
    void op_quantize_linear(const NodeInfo& n);
    void op_ewise_binary(const NodeInfo& n, const std::string& op);
    void op_unary(const NodeInfo& n, const std::string& kname);
    void op_pow(const NodeInfo& n);
    void op_sqrt(const NodeInfo& n);
    void op_clip(const NodeInfo& n);
    void op_max_min(const NodeInfo& n, const std::string& op);
    void op_conv(const NodeInfo& n);
    void op_gemm(const NodeInfo& n);
    void op_matmul(const NodeInfo& n);
    void op_instance_norm(const NodeInfo& n);
    void op_reduce_mean(const NodeInfo& n);
    void op_softmax(const NodeInfo& n);
    void op_reshape(const NodeInfo& n);
    void op_transpose(const NodeInfo& n);
    void op_slice(const NodeInfo& n);
    void op_concat(const NodeInfo& n);

    // Diff helpers
    DiffStats diff_against_golden(const std::string& golden_dir,
                                   const Tensor& got);
    void dump_and_log_tensor(const std::string& tag, const Tensor& t);
};

} // namespace ocl

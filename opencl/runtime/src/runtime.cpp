// runtime.cpp — ONNX→OpenCL runtime engine.
//
// Strategy:
//   - Manifest is loaded; weights uploaded to device as cl_mem buffers.
//   - Each node is dispatched by op_type to a handler that allocates the output
//     buffer, sets kernel args, enqueues, then (optionally) reads back to diff
//     against the ORT golden tensor of the same name.
//   - Layout ops (Reshape/Transpose/Slice/Concat) are computed on the host
//     where convenient (shapes only) or via small index kernels.

#include "runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ocl {

// ---------- error helpers ----------
static void cl_check(cl_int e, const char* what) {
    if (e != CL_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "[OpenCL] %s failed: %d", what, (int)e);
        throw std::runtime_error(buf);
    }
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

static std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    size_t n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(n);
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

static std::string sanitize(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (char c : s) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')
            out.push_back(c);
        else
            out.push_back('_');
    }
    return out;
}

// ---------- OpenCL init ----------
void Runtime::init_opencl() {
    cl_uint nplat;
    cl_check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs");
    if (!nplat) throw std::runtime_error("no OpenCL platforms");
    std::vector<cl_platform_id> plats(nplat);
    cl_check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(2)");
    bool ok = false;
    for (auto p : plats) {
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, &device_, nullptr) == CL_SUCCESS) {
            platform_ = p; ok = true; break;
        }
    }
    if (!ok) throw std::runtime_error("no OpenCL devices");

    char name[256] = {0};
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    log.info("OpenCL device: %s", name);

    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    cl_check(err, "clCreateContext");
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    cl_check(err, "clCreateCommandQueue");
}

void Runtime::compile_kernels(const std::string& cl_source_path) {
    std::string src = read_text(cl_source_path);
    const char* src_ptr = src.c_str();
    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, nullptr, &err);
    cl_check(err, "clCreateProgramWithSource");
    err = clBuildProgram(program_, 1, &device_, "-cl-std=CL1.2 -cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::vector<char> buf(logsz + 1, 0);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logsz, buf.data(), nullptr);
        log.err("Kernel build failed:\n%s", buf.data());
        throw std::runtime_error("kernel build failed");
    }

    // Enumerate kernels we know we'll need; create on demand if missing.
    const char* names[] = {
        "quantize_linear_i8", "dequantize_linear_i8", "dequantize_linear_i8_per_axis",
        "ewise_add_eq", "ewise_sub_eq", "ewise_mul_eq", "ewise_div_eq",
        "ewise_max_eq", "ewise_min_eq",
        "bcast_add", "bcast_sub", "bcast_mul", "bcast_div", "bcast_max", "bcast_min",
        "act_sigmoid", "act_tanh", "act_sin", "act_cos", "act_sqrt",
        "act_pow_scalar", "act_clip_scalar",
        "conv2d_nchw", "gemm_transB", "matmul_batched",
        "instance_norm", "reduce_mean_trailing", "softmax_last_axis",
        "transpose_generic", "gather_index", "copy_chunk",
    };
    for (const char* n : names) {
        cl_int e;
        cl_kernel k = clCreateKernel(program_, n, &e);
        if (e == CL_SUCCESS && k) kernels_[n] = k;
        else log.warn("Kernel '%s' missing (err=%d)", n, (int)e);
    }
    log.info("Compiled %zu kernels", kernels_.size());
}

cl_kernel Runtime::kernel(const std::string& name) {
    auto it = kernels_.find(name);
    if (it == kernels_.end()) throw std::runtime_error("missing kernel: " + name);
    return it->second;
}

// ---------- Manifest + tensor table ----------
void Runtime::load_manifest(const std::string& path) {
    manifest_ = Manifest::load(path);
    log.info("Loaded manifest: %zu nodes, %zu initializers",
             manifest_.nodes.size(), manifest_.initializers.size());

    // Register graph inputs/outputs as placeholders (no device buffer yet).
    for (auto& vi : manifest_.inputs) {
        auto t = std::make_unique<Tensor>();
        t->name = vi.name; t->dtype = vi.dtype; t->shape = vi.shape;
        tensors_[vi.name] = std::move(t);
    }
}

void Runtime::load_initializers() {
    for (auto& ii : manifest_.initializers) {
        std::string full = manifest_.root_dir + "/" + ii.file;
        auto bytes = read_binary(full);
        if (bytes.size() != ii.byte_size) {
            log.warn("init %s: size mismatch (file=%zu manifest=%zu)",
                     ii.name.c_str(), bytes.size(), ii.byte_size);
        }
        Tensor* t = alloc_tensor(ii.name, ii.dtype, ii.shape);
        upload(t->device, bytes.data(), bytes.size());

        // Stash small int64 tensors as host shadow (used by Reshape/Slice).
        if (ii.dtype == DType::INT64 || ii.dtype == DType::INT32 ||
            (t->numel() > 0 && t->numel() <= 16)) {
            t->host_bytes = std::move(bytes);
        }
    }
    log.info("Uploaded %zu initializers", manifest_.initializers.size());
}

cl_mem Runtime::alloc_buffer(size_t bytes, cl_mem_flags flags) {
    cl_int err;
    cl_mem b = clCreateBuffer(context_, flags, bytes == 0 ? 1 : bytes, nullptr, &err);
    cl_check(err, "clCreateBuffer");
    return b;
}

void Runtime::upload(cl_mem buf, const void* data, size_t bytes) {
    if (bytes == 0) return;
    cl_check(clEnqueueWriteBuffer(queue_, buf, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
             "clEnqueueWriteBuffer");
}
void Runtime::download(cl_mem buf, void* data, size_t bytes) {
    if (bytes == 0) return;
    cl_check(clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
             "clEnqueueReadBuffer");
}

Tensor* Runtime::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second.get();
}
Tensor* Runtime::require_tensor(const std::string& name) {
    Tensor* t = get_tensor(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

Tensor* Runtime::alloc_tensor(const std::string& name, DType dtype,
                              const std::vector<int64_t>& shape) {
    auto t = std::make_unique<Tensor>();
    t->name = name;
    t->dtype = dtype;
    t->shape = shape;
    int64_t n = 1;
    for (auto d : shape) { if (d <= 0) { n = 0; break; } n *= d; }
    if (n < 0) n = 0;
    t->byte_size = (size_t)n * (size_t)dtype_size(dtype);
    t->device = alloc_buffer(t->byte_size);
    Tensor* raw = t.get();
    tensors_[name] = std::move(t);
    return raw;
}

void Runtime::set_input_fp32(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const float* data) {
    Tensor* t = get_tensor(name);
    if (!t) throw std::runtime_error("unknown input: " + name);
    if (t->shape != shape) {
        t->shape = shape;
        int64_t n = 1; for (auto d : shape) n *= d;
        t->byte_size = (size_t)n * sizeof(float);
        if (t->device) clReleaseMemObject(t->device);
        t->device = alloc_buffer(t->byte_size);
    } else if (!t->device) {
        t->device = alloc_buffer(t->byte_size);
    }
    upload(t->device, data, t->byte_size);
    log.info("Set input %s shape=%s n=%lld", name.c_str(), t->shape_str().c_str(),
             (long long)t->numel());
}

void Runtime::read_output_fp32(const std::string& name, std::vector<float>& out) {
    Tensor* t = require_tensor(name);
    int64_t n = t->numel();
    out.resize((size_t)n);
    download(t->device, out.data(), (size_t)n * sizeof(float));
}

void Runtime::cleanup() {
    for (auto& kv : tensors_) if (kv.second && kv.second->device) clReleaseMemObject(kv.second->device);
    tensors_.clear();
    for (auto& kv : kernels_) clReleaseKernel(kv.second);
    kernels_.clear();
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
    if (queue_)   { clReleaseCommandQueue(queue_); queue_ = nullptr; }
    if (context_) { clReleaseContext(context_); context_ = nullptr; }
}

// ---------- Dispatch ----------
void Runtime::run(const std::string& golden_dir, double abs_tol, double rel_tol,
                  bool continue_on_error) {
    log.info("=== Running %zu nodes ===", manifest_.nodes.size());
    int first_failure = -1;
    for (auto& n : manifest_.nodes) {
        try {
            run_node(n);
            ops_done_++;
        } catch (const std::exception& e) {
            ops_failed_++;
            log.err("node #%d %s '%s' FAILED: %s",
                    n.index, n.op_type.c_str(), n.name.c_str(), e.what());
            if (first_failure < 0) first_failure = n.index;
            if (!continue_on_error) break;
            continue;
        }
        if (!golden_dir.empty()) {
            for (auto& outname : n.outputs) {
                Tensor* t = get_tensor(outname);
                if (!t || t->dtype != DType::FLOAT32) continue;
                DiffStats ds = diff_against_golden(golden_dir, *t);
                if (!ds.found_golden) continue;
                bool fail = ds.max_abs_err > abs_tol;
                log.info("  diff #%d %s out=%s max_abs=%.3e mean_abs=%.3e %s",
                         n.index, n.op_type.c_str(), outname.c_str(),
                         ds.max_abs_err, ds.mean_abs_err,
                         fail ? "FAIL" : "ok");
                if (fail && !continue_on_error) {
                    dump_and_log_tensor("fail", *t);
                    throw std::runtime_error("golden diff failed at node " +
                                             std::to_string(n.index));
                }
            }
        }
    }
    log.info("=== Done. ok=%d failed=%d skipped=%d (first_failure=%d) ===",
             ops_done_, ops_failed_, ops_skipped_, first_failure);
}

DiffStats Runtime::diff_against_golden(const std::string& golden_dir, const Tensor& got) {
    DiffStats s;
    std::string path = golden_dir + "/" + sanitize(got.name) + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return s;
    s.found_golden = true;

    int64_t n = got.numel();
    std::vector<float> gold((size_t)n);
    f.read(reinterpret_cast<char*>(gold.data()), n * sizeof(float));
    std::vector<float> mine((size_t)n);
    download(got.device, mine.data(), (size_t)n * sizeof(float));

    double sum_abs = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs((double)mine[i] - (double)gold[i]);
        sum_abs += d;
        if (d > s.max_abs_err) {
            s.max_abs_err = d;
            s.argmax = i;
            s.golden_at_argmax = gold[i];
            s.got_at_argmax = mine[i];
        }
        double denom = std::fabs((double)gold[i]) + 1e-9;
        double rel = d / denom;
        if (rel > s.max_rel_err) s.max_rel_err = rel;
    }
    s.mean_abs_err = (n > 0) ? sum_abs / (double)n : 0.0;
    return s;
}

void Runtime::dump_and_log_tensor(const std::string& tag, const Tensor& t) {
    int64_t n = t.numel();
    if (n <= 0 || t.dtype != DType::FLOAT32) return;
    std::vector<float> host((size_t)n);
    download(t.device, host.data(), (size_t)n * sizeof(float));
    log.dump_fp32(tag, t.name, t.shape, host.data(), (size_t)n);
}

// ---------- Per-node dispatch ----------
void Runtime::run_node(const NodeInfo& n) {
    const std::string& op = n.op_type;
    if      (op == "DequantizeLinear")   op_dequantize_linear(n);
    else if (op == "QuantizeLinear")     op_quantize_linear(n);
    else if (op == "Add")                op_ewise_binary(n, "add");
    else if (op == "Sub")                op_ewise_binary(n, "sub");
    else if (op == "Mul")                op_ewise_binary(n, "mul");
    else if (op == "Div")                op_ewise_binary(n, "div");
    else if (op == "Max")                op_max_min(n, "max");
    else if (op == "Min")                op_max_min(n, "min");
    else if (op == "Sigmoid")            op_unary(n, "act_sigmoid");
    else if (op == "Tanh")               op_unary(n, "act_tanh");
    else if (op == "Sin")                op_unary(n, "act_sin");
    else if (op == "Cos")                op_unary(n, "act_cos");
    else if (op == "Sqrt")               op_unary(n, "act_sqrt");
    else if (op == "Pow")                op_pow(n);
    else if (op == "Clip")               op_clip(n);
    else if (op == "Conv")               op_conv(n);
    else if (op == "Gemm")               op_gemm(n);
    else if (op == "MatMul")             op_matmul(n);
    else if (op == "InstanceNormalization") op_instance_norm(n);
    else if (op == "ReduceMean")         op_reduce_mean(n);
    else if (op == "Softmax")            op_softmax(n);
    else if (op == "Reshape")            op_reshape(n);
    else if (op == "Transpose")          op_transpose(n);
    else if (op == "Slice")              op_slice(n);
    else if (op == "Concat")             op_concat(n);
    else {
        ops_skipped_++;
        log.warn("[skip] node #%d unsupported op %s '%s'", n.index, op.c_str(), n.name.c_str());
        // Allocate zeroed outputs so downstream lookups don't fail.
        for (auto& outn : n.outputs) {
            if (outn.empty() || get_tensor(outn)) continue;
            alloc_tensor(outn, DType::FLOAT32, {1});
        }
    }
}

// ---------- Op implementations ----------

// Helper: enqueue 1D kernel.
static void enqueue_1d(cl_command_queue q, cl_kernel k, size_t global) {
    if (global == 0) return;
    size_t g = ((global + 63) / 64) * 64;
    cl_check(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
             "clEnqueueNDRangeKernel(1d)");
}

void Runtime::op_dequantize_linear(const NodeInfo& n) {
    // y = (x - zp) * scale
    Tensor* x  = require_tensor(n.inputs[0]);
    Tensor* sc = require_tensor(n.inputs[1]);
    Tensor* zp = require_tensor(n.inputs[2]);
    auto& outn = n.outputs[0];
    Tensor* y = alloc_tensor(outn, DType::FLOAT32, x->shape);
    int64_t nelem = x->numel();

    // Pull scalar scale/zp to host (these are tiny). Per-axis path is rare in our QDQ.
    if (sc->numel() == 1 && zp->numel() == 1) {
        float scale_val = 0.0f;
        download(sc->device, &scale_val, sizeof(float));
        int zp_val = 0;
        if (zp->dtype == DType::INT8) {
            int8_t v; download(zp->device, &v, 1); zp_val = (int)v;
        } else if (zp->dtype == DType::UINT8) {
            uint8_t v; download(zp->device, &v, 1); zp_val = (int)v;
        } else {
            // Best-effort
            int32_t v = 0; download(zp->device, &v, dtype_size(zp->dtype)); zp_val = v;
        }
        cl_kernel k = kernel("dequantize_linear_i8");
        int nn = (int)nelem;
        clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
        clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
        clSetKernelArg(k, 2, sizeof(float),  &scale_val);
        clSetKernelArg(k, 3, sizeof(int),    &zp_val);
        clSetKernelArg(k, 4, sizeof(int),    &nn);
        enqueue_1d(queue_, k, nelem);
    } else {
        // Per-axis: assume axis = 0 (out-channel for conv weights).
        cl_kernel k = kernel("dequantize_linear_i8_per_axis");
        int axis_dim = (int)x->shape[0];
        int inner = 1;
        for (size_t d = 1; d < x->shape.size(); ++d) inner *= (int)x->shape[d];
        int nn = (int)nelem;
        clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
        clSetKernelArg(k, 1, sizeof(cl_mem), &sc->device);
        clSetKernelArg(k, 2, sizeof(cl_mem), &zp->device);
        clSetKernelArg(k, 3, sizeof(cl_mem), &y->device);
        clSetKernelArg(k, 4, sizeof(int),    &axis_dim);
        clSetKernelArg(k, 5, sizeof(int),    &inner);
        clSetKernelArg(k, 6, sizeof(int),    &nn);
        enqueue_1d(queue_, k, nelem);
    }
}

void Runtime::op_quantize_linear(const NodeInfo& n) {
    Tensor* x  = require_tensor(n.inputs[0]);
    Tensor* sc = require_tensor(n.inputs[1]);
    Tensor* zp = require_tensor(n.inputs[2]);
    auto& outn = n.outputs[0];
    Tensor* y = alloc_tensor(outn, zp->dtype, x->shape);

    float scale_val = 0.0f;
    download(sc->device, &scale_val, sizeof(float));
    int zp_val = 0;
    if (zp->dtype == DType::INT8) {
        int8_t v; download(zp->device, &v, 1); zp_val = (int)v;
    } else if (zp->dtype == DType::UINT8) {
        uint8_t v; download(zp->device, &v, 1); zp_val = (int)v;
    }
    cl_kernel k = kernel("quantize_linear_i8");
    int nn = (int)x->numel();
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 2, sizeof(float),  &scale_val);
    clSetKernelArg(k, 3, sizeof(int),    &zp_val);
    clSetKernelArg(k, 4, sizeof(int),    &nn);
    enqueue_1d(queue_, k, x->numel());
}

// ---------- Broadcasting helpers ----------
static std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                            const std::vector<int64_t>& b) {
    size_t r = std::max(a.size(), b.size());
    std::vector<int64_t> out(r);
    for (size_t i = 0; i < r; ++i) {
        int64_t ad = (i < r - a.size()) ? 1 : a[i - (r - a.size())];
        int64_t bd = (i < r - b.size()) ? 1 : b[i - (r - b.size())];
        if (ad == bd) out[i] = ad;
        else if (ad == 1) out[i] = bd;
        else if (bd == 1) out[i] = ad;
        else throw std::runtime_error("incompatible broadcast");
    }
    return out;
}

static std::vector<int> compute_broadcast_strides(const std::vector<int64_t>& src,
                                                  const std::vector<int64_t>& out) {
    // Pad src on the left with 1s, then compute strides such that broadcast
    // axes (src dim == 1) get stride 0.
    size_t r = out.size();
    std::vector<int64_t> padded(r, 1);
    for (size_t i = 0; i < src.size(); ++i)
        padded[r - src.size() + i] = src[i];
    // Compute "natural" strides over padded.
    std::vector<int> str(r, 0);
    int s = 1;
    for (int i = (int)r - 1; i >= 0; --i) {
        if (padded[i] == out[i]) { str[i] = s; }
        else { str[i] = 0; }  // broadcast axis
        s *= (int)padded[i];
    }
    return str;
}

void Runtime::op_ewise_binary(const NodeInfo& n, const std::string& op) {
    Tensor* a = require_tensor(n.inputs[0]);
    Tensor* b = require_tensor(n.inputs[1]);
    auto& outn = n.outputs[0];
    if (a->shape == b->shape) {
        Tensor* y = alloc_tensor(outn, DType::FLOAT32, a->shape);
        cl_kernel k = kernel("ewise_" + op + "_eq");
        int nn = (int)a->numel();
        clSetKernelArg(k, 0, sizeof(cl_mem), &a->device);
        clSetKernelArg(k, 1, sizeof(cl_mem), &b->device);
        clSetKernelArg(k, 2, sizeof(cl_mem), &y->device);
        clSetKernelArg(k, 3, sizeof(int),    &nn);
        enqueue_1d(queue_, k, a->numel());
        return;
    }
    auto outs = broadcast_shape(a->shape, b->shape);
    Tensor* y = alloc_tensor(outn, DType::FLOAT32, outs);
    auto as = compute_broadcast_strides(a->shape, outs);
    auto bs = compute_broadcast_strides(b->shape, outs);
    std::vector<int> outshape(outs.begin(), outs.end());
    int rank = (int)outs.size();
    if (rank > 8) throw std::runtime_error("rank > 8 unsupported");

    cl_int err;
    cl_mem shape_buf  = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       outshape.size() * sizeof(int), outshape.data(), &err);
    cl_check(err, "buf shape");
    cl_mem astr_buf   = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       as.size() * sizeof(int), as.data(), &err); cl_check(err, "buf a_str");
    cl_mem bstr_buf   = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       bs.size() * sizeof(int), bs.data(), &err); cl_check(err, "buf b_str");
    cl_kernel k = kernel("bcast_" + op);
    int nout = (int)y->numel();
    clSetKernelArg(k, 0, sizeof(cl_mem), &a->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b->device);
    clSetKernelArg(k, 2, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 3, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(k, 4, sizeof(cl_mem), &astr_buf);
    clSetKernelArg(k, 5, sizeof(cl_mem), &bstr_buf);
    clSetKernelArg(k, 6, sizeof(int),    &rank);
    clSetKernelArg(k, 7, sizeof(int),    &nout);
    enqueue_1d(queue_, k, (size_t)nout);
    clFinish(queue_);
    clReleaseMemObject(shape_buf);
    clReleaseMemObject(astr_buf);
    clReleaseMemObject(bstr_buf);
}

void Runtime::op_max_min(const NodeInfo& n, const std::string& op) {
    op_ewise_binary(n, op);
}

void Runtime::op_unary(const NodeInfo& n, const std::string& kname) {
    Tensor* x = require_tensor(n.inputs[0]);
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
    cl_kernel k = kernel(kname);
    int nn = (int)x->numel();
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 2, sizeof(int),    &nn);
    enqueue_1d(queue_, k, x->numel());
}

void Runtime::op_pow(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    Tensor* p = require_tensor(n.inputs[1]);
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
    if (p->numel() != 1) {
        log.warn("Pow with non-scalar exponent not implemented; using element-wise broadcast may be needed.");
        return op_ewise_binary(n, "mul");  // best-effort placeholder
    }
    float pv; download(p->device, &pv, sizeof(float));
    cl_kernel k = kernel("act_pow_scalar");
    int nn = (int)x->numel();
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(float),  &pv);
    clSetKernelArg(k, 2, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 3, sizeof(int),    &nn);
    enqueue_1d(queue_, k, x->numel());
}

void Runtime::op_clip(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    float lo = -1e30f, hi = 1e30f;
    if (n.inputs.size() > 1 && !n.inputs[1].empty()) {
        Tensor* t = require_tensor(n.inputs[1]); download(t->device, &lo, sizeof(float));
    }
    if (n.inputs.size() > 2 && !n.inputs[2].empty()) {
        Tensor* t = require_tensor(n.inputs[2]); download(t->device, &hi, sizeof(float));
    }
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
    cl_kernel k = kernel("act_clip_scalar");
    int nn = (int)x->numel();
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(float),  &lo);
    clSetKernelArg(k, 2, sizeof(float),  &hi);
    clSetKernelArg(k, 3, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 4, sizeof(int),    &nn);
    enqueue_1d(queue_, k, x->numel());
}

// ---------- Conv / Gemm / MatMul ----------
void Runtime::op_conv(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);          // [N, IC, IH, IW]
    Tensor* w = require_tensor(n.inputs[1]);          // [OC, IC, KH, KW]
    Tensor* b = (n.inputs.size() > 2 && !n.inputs[2].empty()) ? require_tensor(n.inputs[2]) : nullptr;
    auto* kshape  = n.attr("kernel_shape");
    auto* pads    = n.attr("pads");
    auto* strides = n.attr("strides");
    auto* dils    = n.attr("dilations");

    int N = (int)x->shape[0], IC = (int)x->shape[1], IH = (int)x->shape[2], IW = (int)x->shape[3];
    int OC = (int)w->shape[0];
    int KH = kshape ? (int)kshape->ints[0] : (int)w->shape[2];
    int KW = kshape ? (int)kshape->ints[1] : (int)w->shape[3];
    int PT = pads    ? (int)pads->ints[0] : 0;
    int PL = pads    ? (int)pads->ints[1] : 0;
    int PB = pads    ? (int)pads->ints[2] : PT;
    int PR = pads    ? (int)pads->ints[3] : PL;
    int SH = strides ? (int)strides->ints[0] : 1;
    int SW = strides ? (int)strides->ints[1] : 1;
    int DH = dils    ? (int)dils->ints[0] : 1;
    int DW = dils    ? (int)dils->ints[1] : 1;
    int OH = (IH + PT + PB - DH * (KH - 1) - 1) / SH + 1;
    int OW = (IW + PL + PR - DW * (KW - 1) - 1) / SW + 1;

    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32,
                             {(int64_t)N, (int64_t)OC, (int64_t)OH, (int64_t)OW});
    cl_kernel k = kernel("conv2d_nchw");
    int has_bias = b ? 1 : 0;
    cl_mem bias_buf = b ? b->device : x->device;  // dummy when has_bias=0
    int ai = 0;
    clSetKernelArg(k, ai++, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &w->device);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &bias_buf);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, ai++, sizeof(int), &N);
    clSetKernelArg(k, ai++, sizeof(int), &IC);
    clSetKernelArg(k, ai++, sizeof(int), &IH);
    clSetKernelArg(k, ai++, sizeof(int), &IW);
    clSetKernelArg(k, ai++, sizeof(int), &OC);
    clSetKernelArg(k, ai++, sizeof(int), &KH);
    clSetKernelArg(k, ai++, sizeof(int), &KW);
    clSetKernelArg(k, ai++, sizeof(int), &PT);
    clSetKernelArg(k, ai++, sizeof(int), &PL);
    clSetKernelArg(k, ai++, sizeof(int), &SH);
    clSetKernelArg(k, ai++, sizeof(int), &SW);
    clSetKernelArg(k, ai++, sizeof(int), &DH);
    clSetKernelArg(k, ai++, sizeof(int), &DW);
    clSetKernelArg(k, ai++, sizeof(int), &OH);
    clSetKernelArg(k, ai++, sizeof(int), &OW);
    clSetKernelArg(k, ai++, sizeof(int), &has_bias);
    size_t gws[3] = { (size_t)OW, (size_t)OH, (size_t)(N * OC) };
    // Round up to 8 for x/y
    auto roundup = [](size_t v, size_t m) { return ((v + m - 1) / m) * m; };
    gws[0] = roundup(gws[0], 8); gws[1] = roundup(gws[1], 8);
    cl_check(clEnqueueNDRangeKernel(queue_, k, 3, nullptr, gws, nullptr, 0, nullptr, nullptr),
             "conv NDRange");
}

void Runtime::op_gemm(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    Tensor* w = require_tensor(n.inputs[1]);
    Tensor* b = (n.inputs.size() > 2 && !n.inputs[2].empty()) ? require_tensor(n.inputs[2]) : nullptr;

    int transB = 1;
    if (auto* a = n.attr("transB")) transB = (int)a->i;
    if (transB != 1) {
        log.warn("Gemm with transB=%d not implemented; assuming W stored as [N,K]", transB);
    }
    int M = (int)x->shape[0];
    int K = (int)x->shape[1];
    int N = (int)w->shape[0];  // since transB=1: W is [N,K]
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, {(int64_t)M, (int64_t)N});

    cl_kernel k = kernel("gemm_transB");
    cl_mem bias_buf = b ? b->device : x->device;
    int has_bias = b ? 1 : 0;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &w->device);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bias_buf);
    clSetKernelArg(k, 3, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 4, sizeof(int), &M);
    clSetKernelArg(k, 5, sizeof(int), &N);
    clSetKernelArg(k, 6, sizeof(int), &K);
    clSetKernelArg(k, 7, sizeof(int), &has_bias);
    size_t g[2] = { ((size_t)N + 7) / 8 * 8, ((size_t)M + 7) / 8 * 8 };
    cl_check(clEnqueueNDRangeKernel(queue_, k, 2, nullptr, g, nullptr, 0, nullptr, nullptr),
             "gemm NDRange");
}

void Runtime::op_matmul(const NodeInfo& n) {
    Tensor* a = require_tensor(n.inputs[0]);
    Tensor* b = require_tensor(n.inputs[1]);
    // Treat last two dims as M, K  and  K, N. Everything before is the batch.
    if (a->shape.size() < 2 || b->shape.size() < 2)
        throw std::runtime_error("MatMul requires rank >= 2");
    int ra = (int)a->shape.size(), rb = (int)b->shape.size();
    int M = (int)a->shape[ra - 2], K = (int)a->shape[ra - 1];
    int Kb = (int)b->shape[rb - 2], N = (int)b->shape[rb - 1];
    if (K != Kb) throw std::runtime_error("MatMul K mismatch");
    // Batch is product of leading dims (must match; no broadcasting supported yet).
    int64_t Ba = 1; for (int i = 0; i < ra - 2; ++i) Ba *= a->shape[i];
    int64_t Bb = 1; for (int i = 0; i < rb - 2; ++i) Bb *= b->shape[i];
    if (Ba != Bb) throw std::runtime_error("MatMul batch broadcast not supported");
    int B = (int)Ba;

    auto outshape = a->shape;
    outshape[outshape.size() - 1] = N;
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, outshape);

    cl_kernel k = kernel("matmul_batched");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b->device);
    clSetKernelArg(k, 2, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 3, sizeof(int), &B);
    clSetKernelArg(k, 4, sizeof(int), &M);
    clSetKernelArg(k, 5, sizeof(int), &N);
    clSetKernelArg(k, 6, sizeof(int), &K);
    auto roundup = [](size_t v, size_t m) { return ((v + m - 1) / m) * m; };
    size_t g[3] = { roundup((size_t)N, 8), roundup((size_t)M, 8), (size_t)B };
    cl_check(clEnqueueNDRangeKernel(queue_, k, 3, nullptr, g, nullptr, 0, nullptr, nullptr),
             "matmul NDRange");
}

// ---------- Normalisation / Reduction / Softmax ----------
void Runtime::op_instance_norm(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    Tensor* scale = require_tensor(n.inputs[1]);
    Tensor* bias  = require_tensor(n.inputs[2]);
    float eps = 1e-5f;
    if (auto* a = n.attr("epsilon")) eps = (float)a->f;

    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
    int N = (int)x->shape[0], C = (int)x->shape[1];
    int S = 1; for (size_t d = 2; d < x->shape.size(); ++d) S *= (int)x->shape[d];

    cl_kernel k = kernel("instance_norm");
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &scale->device);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bias->device);
    clSetKernelArg(k, 3, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 4, sizeof(int), &N);
    clSetKernelArg(k, 5, sizeof(int), &C);
    clSetKernelArg(k, 6, sizeof(int), &S);
    clSetKernelArg(k, 7, sizeof(float), &eps);
    auto roundup = [](size_t v, size_t m) { return ((v + m - 1) / m) * m; };
    size_t g[2] = { roundup((size_t)C, 8), (size_t)N };
    cl_check(clEnqueueNDRangeKernel(queue_, k, 2, nullptr, g, nullptr, 0, nullptr, nullptr),
             "instancenorm NDRange");
}

void Runtime::op_reduce_mean(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    std::vector<int64_t> axes;
    if (auto* a = n.attr("axes")) for (auto v : a->ints) axes.push_back(v);
    int rank = (int)x->shape.size();
    for (auto& a : axes) if (a < 0) a += rank;
    std::sort(axes.begin(), axes.end());
    // Common case: axes are trailing.
    bool trailing = true;
    for (int i = 0; i < (int)axes.size(); ++i)
        if (axes[axes.size() - 1 - i] != rank - 1 - i) { trailing = false; break; }
    if (!trailing) {
        log.warn("ReduceMean non-trailing axes not yet implemented; node %s", n.name.c_str());
        // Allocate zeroed output so dispatch keeps going.
        std::vector<int64_t> outs = x->shape;
        for (auto a : axes) outs[a] = 1;
        alloc_tensor(n.outputs[0], DType::FLOAT32, outs);
        ops_skipped_++;
        return;
    }
    int inner = 1; for (auto a : axes) inner *= (int)x->shape[a];
    int outer = (int)(x->numel() / inner);
    std::vector<int64_t> outs = x->shape;
    for (auto a : axes) outs[a] = 1;
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, outs);

    cl_kernel k = kernel("reduce_mean_trailing");
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 2, sizeof(int), &outer);
    clSetKernelArg(k, 3, sizeof(int), &inner);
    enqueue_1d(queue_, k, (size_t)outer);
}

void Runtime::op_softmax(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    int axis = -1;
    if (auto* a = n.attr("axis")) axis = (int)a->i;
    int rank = (int)x->shape.size();
    if (axis < 0) axis += rank;
    if (axis != rank - 1) {
        log.warn("Softmax non-trailing axis (%d) not yet implemented; node %s",
                 axis, n.name.c_str());
        alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
        ops_skipped_++;
        return;
    }
    int inner = (int)x->shape[rank - 1];
    int outer = (int)(x->numel() / inner);
    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, x->shape);
    cl_kernel k = kernel("softmax_last_axis");
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 2, sizeof(int), &outer);
    clSetKernelArg(k, 3, sizeof(int), &inner);
    enqueue_1d(queue_, k, (size_t)outer);
}

// ---------- Layout ops ----------
void Runtime::op_reshape(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    Tensor* sh = require_tensor(n.inputs[1]);
    if (sh->host_bytes.empty()) {
        sh->host_bytes.resize(sh->byte_size);
        download(sh->device, sh->host_bytes.data(), sh->byte_size);
    }
    int n_dims = (int)(sh->host_bytes.size() / sizeof(int64_t));
    const int64_t* sh_arr = reinterpret_cast<const int64_t*>(sh->host_bytes.data());
    std::vector<int64_t> outshape(sh_arr, sh_arr + n_dims);

    int64_t nelem = x->numel();
    int infer_idx = -1; int64_t known = 1;
    for (int i = 0; i < n_dims; ++i) {
        if (outshape[i] == -1) { infer_idx = i; }
        else if (outshape[i] == 0) outshape[i] = x->shape[i];
        else known *= outshape[i];
    }
    if (infer_idx >= 0) outshape[infer_idx] = nelem / known;

    // Allocate new output buffer and copy bytes verbatim (NCHW layout assumed).
    Tensor* y = alloc_tensor(n.outputs[0], x->dtype, outshape);
    cl_check(clEnqueueCopyBuffer(queue_, x->device, y->device, 0, 0,
                                 x->byte_size, 0, nullptr, nullptr), "reshape copy");
}

void Runtime::op_transpose(const NodeInfo& n) {
    Tensor* x = require_tensor(n.inputs[0]);
    auto* perm_attr = n.attr("perm");
    int rank = (int)x->shape.size();
    std::vector<int> perm(rank);
    if (perm_attr) for (int i = 0; i < rank; ++i) perm[i] = (int)perm_attr->ints[i];
    else for (int i = 0; i < rank; ++i) perm[i] = rank - 1 - i;

    std::vector<int64_t> outshape(rank);
    for (int i = 0; i < rank; ++i) outshape[i] = x->shape[perm[i]];

    // input strides
    std::vector<int> instride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) instride[i] = instride[i + 1] * (int)x->shape[i + 1];
    // For each output axis d, the stride into x is instride[perm[d]].
    std::vector<int> out_to_in(rank);
    for (int d = 0; d < rank; ++d) out_to_in[d] = instride[perm[d]];

    Tensor* y = alloc_tensor(n.outputs[0], x->dtype, outshape);
    std::vector<int> outshape_i(outshape.begin(), outshape.end());
    cl_int err;
    cl_mem shape_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      outshape_i.size() * sizeof(int), outshape_i.data(), &err);
    cl_check(err, "transpose shape buf");
    cl_mem stride_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       out_to_in.size() * sizeof(int), out_to_in.data(), &err);
    cl_check(err, "transpose stride buf");
    int nout = (int)y->numel();
    cl_kernel k = kernel("transpose_generic");
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 2, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(k, 3, sizeof(cl_mem), &stride_buf);
    clSetKernelArg(k, 4, sizeof(int), &rank);
    clSetKernelArg(k, 5, sizeof(int), &nout);
    enqueue_1d(queue_, k, (size_t)nout);
    clFinish(queue_);
    clReleaseMemObject(shape_buf);
    clReleaseMemObject(stride_buf);
}

void Runtime::op_slice(const NodeInfo& n) {
    // ONNX Slice (opset >= 10): data, starts, ends, [axes], [steps]
    Tensor* x = require_tensor(n.inputs[0]);
    auto get_i64_vec = [&](const std::string& name, const std::vector<int64_t>& dflt) {
        if (name.empty()) return dflt;
        Tensor* t = require_tensor(name);
        if (t->host_bytes.empty()) {
            t->host_bytes.resize(t->byte_size);
            download(t->device, t->host_bytes.data(), t->byte_size);
        }
        if (t->dtype == DType::INT64) {
            const int64_t* p = reinterpret_cast<const int64_t*>(t->host_bytes.data());
            return std::vector<int64_t>(p, p + t->numel());
        } else if (t->dtype == DType::INT32) {
            const int32_t* p = reinterpret_cast<const int32_t*>(t->host_bytes.data());
            std::vector<int64_t> v; v.reserve(t->numel());
            for (int64_t i = 0; i < t->numel(); ++i) v.push_back(p[i]);
            return v;
        }
        return dflt;
    };
    auto starts = get_i64_vec(n.inputs[1], {});
    auto ends   = get_i64_vec(n.inputs[2], {});
    std::vector<int64_t> def_axes;
    for (int i = 0; i < (int)starts.size(); ++i) def_axes.push_back(i);
    auto axes  = (n.inputs.size() > 3 && !n.inputs[3].empty())
                  ? get_i64_vec(n.inputs[3], def_axes) : def_axes;
    auto steps = (n.inputs.size() > 4 && !n.inputs[4].empty())
                  ? get_i64_vec(n.inputs[4], std::vector<int64_t>(starts.size(), 1))
                  : std::vector<int64_t>(starts.size(), 1);

    int rank = (int)x->shape.size();
    std::vector<int64_t> outshape = x->shape;
    std::vector<int64_t> ax_start(rank, 0), ax_step(rank, 1);
    for (int i = 0; i < rank; ++i) ax_step[i] = 1;
    for (size_t k = 0; k < axes.size(); ++k) {
        int ax = (int)axes[k]; if (ax < 0) ax += rank;
        int64_t st = starts[k], en = ends[k], stp = steps[k];
        if (st < 0) st += x->shape[ax];
        if (en < 0) en += x->shape[ax];
        if (st < 0) st = 0;
        if (en > x->shape[ax]) en = x->shape[ax];
        int64_t cnt = (en - st + (stp > 0 ? (stp - 1) : (stp + 1))) / stp;
        if (cnt < 0) cnt = 0;
        outshape[ax] = cnt;
        ax_start[ax] = st;
        ax_step[ax] = stp;
    }
    Tensor* y = alloc_tensor(n.outputs[0], x->dtype, outshape);

    // Build an index list on the host (works for any rank/strides; not the fastest).
    int64_t total = y->numel();
    std::vector<int> src_idx((size_t)total);
    std::vector<int> instride(rank, 1);
    for (int i = rank - 2; i >= 0; --i) instride[i] = instride[i + 1] * (int)x->shape[i + 1];
    std::vector<int64_t> dim_size(outshape);
    for (int64_t i = 0; i < total; ++i) {
        int64_t rem = i;
        int64_t suffix = 1;
        std::vector<int64_t> coord(rank, 0);
        for (int d = rank - 1; d >= 0; --d) {
            coord[d] = (rem / suffix) % dim_size[d];
            suffix *= dim_size[d];
        }
        int64_t src = 0;
        for (int d = 0; d < rank; ++d) src += (ax_start[d] + coord[d] * ax_step[d]) * instride[d];
        src_idx[(size_t)i] = (int)src;
    }
    cl_int err;
    cl_mem idx_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    src_idx.size() * sizeof(int), src_idx.data(), &err);
    cl_check(err, "slice idx buf");
    cl_kernel k = kernel("gather_index");
    int nout = (int)total;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x->device);
    clSetKernelArg(k, 1, sizeof(cl_mem), &idx_buf);
    clSetKernelArg(k, 2, sizeof(cl_mem), &y->device);
    clSetKernelArg(k, 3, sizeof(int), &nout);
    enqueue_1d(queue_, k, (size_t)nout);
    clFinish(queue_);
    clReleaseMemObject(idx_buf);
}

void Runtime::op_concat(const NodeInfo& n) {
    int axis = 0;
    if (auto* a = n.attr("axis")) axis = (int)a->i;
    std::vector<Tensor*> inputs;
    for (auto& in : n.inputs) inputs.push_back(require_tensor(in));
    int rank = (int)inputs[0]->shape.size();
    if (axis < 0) axis += rank;
    std::vector<int64_t> outshape = inputs[0]->shape;
    int64_t axis_total = 0;
    for (auto* t : inputs) axis_total += t->shape[axis];
    outshape[axis] = axis_total;

    Tensor* y = alloc_tensor(n.outputs[0], DType::FLOAT32, outshape);
    // We iterate over outer (product of dims before axis) and for each, copy
    // each input's slab (axis_dim * inner) into the right offset.
    int64_t outer = 1;
    for (int d = 0; d < axis; ++d) outer *= outshape[d];
    int64_t inner = 1;
    for (int d = axis + 1; d < rank; ++d) inner *= outshape[d];
    int64_t out_axis = outshape[axis];

    cl_kernel k = kernel("copy_chunk");
    int64_t axis_offset = 0;
    for (auto* t : inputs) {
        int64_t a_dim = t->shape[axis];
        int64_t chunk_count = a_dim * inner;
        for (int64_t o = 0; o < outer; ++o) {
            int src_off = (int)(o * a_dim * inner);
            int dst_off = (int)(o * out_axis * inner + axis_offset * inner);
            int nn = (int)chunk_count;
            clSetKernelArg(k, 0, sizeof(cl_mem), &t->device);
            clSetKernelArg(k, 1, sizeof(cl_mem), &y->device);
            clSetKernelArg(k, 2, sizeof(int), &src_off);
            clSetKernelArg(k, 3, sizeof(int), &dst_off);
            clSetKernelArg(k, 4, sizeof(int), &nn);
            enqueue_1d(queue_, k, (size_t)chunk_count);
        }
        axis_offset += a_dim;
    }
}

} // namespace ocl

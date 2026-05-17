#pragma once
// Structured per-op logger. Writes plaintext .log + binary tensor dumps under
// a run-id subdir, so every op is traceable post-mortem.

#include "tensor.hpp"
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace ocl {

inline void mkdir_p(const std::string& path) {
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '/') {
            if (!acc.empty()) mkdir(acc.c_str(), 0755);
        }
        acc.push_back(path[i]);
    }
    if (!acc.empty()) mkdir(acc.c_str(), 0755);
}

class Logger {
public:
    Logger() {}

    void open(const std::string& base_dir) {
        // Run-id from timestamp.
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
        run_id_ = buf;
        run_dir_ = base_dir + "/run_" + run_id_;
        mkdir_p(run_dir_ + "/tensors");
        log_path_ = run_dir_ + "/run.log";
        log_.open(log_path_, std::ios::out | std::ios::trunc);
        if (!log_.is_open()) {
            std::cerr << "[Logger] WARN: could not open " << log_path_ << "\n";
        }
        info("Logger opened: run_dir=%s", run_dir_.c_str());
    }

    void close() {
        if (log_.is_open()) log_.close();
    }

    const std::string& run_dir() const { return run_dir_; }
    const std::string& run_id()  const { return run_id_; }

    void logf(const char* level, const char* fmt, va_list ap) {
        char buf[2048];
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        std::string line = std::string("[") + level + "] " + buf;
        if (log_.is_open()) { log_ << line << "\n"; log_.flush(); }
        if (echo_) std::cout << line << "\n";
    }

    void info(const char* fmt, ...) { va_list ap; va_start(ap, fmt); logf("INFO", fmt, ap); va_end(ap); }
    void warn(const char* fmt, ...) { va_list ap; va_start(ap, fmt); logf("WARN", fmt, ap); va_end(ap); }
    void err (const char* fmt, ...) { va_list ap; va_start(ap, fmt); logf("ERR ", fmt, ap); va_end(ap); }
    void dbg (const char* fmt, ...) { if (!verbose_) return; va_list ap; va_start(ap, fmt); logf("DBG ", fmt, ap); va_end(ap); }

    void set_echo(bool e) { echo_ = e; }
    void set_verbose(bool v) { verbose_ = v; }

    // Save a host-side fp32 buffer with metadata for post-mortem diff.
    void dump_fp32(const std::string& tag, const std::string& tensor_name,
                   const std::vector<int64_t>& shape, const float* data, size_t n) {
        std::string safe = sanitize(tensor_name);
        std::string bin = run_dir_ + "/tensors/" + tag + "_" + safe + ".bin";
        std::ofstream f(bin, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
        // Stats line
        float mn = 0, mx = 0, sum = 0;
        if (n) { mn = mx = data[0]; for (size_t i = 0; i < n; ++i) {
                    float v = data[i]; mn = v < mn ? v : mn; mx = v > mx ? v : mx; sum += v; } }
        float mean = n ? sum / float(n) : 0.0f;
        info("  dump %s shape=%s n=%zu min=%.4f max=%.4f mean=%.4f → %s",
             tensor_name.c_str(), shape_str(shape).c_str(), n, mn, mx, mean, bin.c_str());
    }

private:
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
    static std::string shape_str(const std::vector<int64_t>& s) {
        std::string out = "[";
        for (size_t i = 0; i < s.size(); ++i) {
            out += std::to_string(s[i]);
            if (i + 1 < s.size()) out += ",";
        }
        return out + "]";
    }

    std::ofstream log_;
    std::string run_dir_;
    std::string run_id_;
    std::string log_path_;
    bool echo_   = true;
    bool verbose_ = false;
};

} // namespace ocl

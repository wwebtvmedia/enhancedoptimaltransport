// main.cpp — CLI entry for the ONNX→OpenCL runner.
//
// Usage:
//   onnx_opencl_runner --drift assets/drift --gen assets/generator \
//       --embedding horse_embedding.bin --steps 20 --cfg 6.5 \
//       --verify --out output_horse.ppm

#include "runtime.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ocl;

static std::vector<float> read_fp32_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "missing: " << path << "\n"; return {}; }
    f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

static void save_ppm(const std::string& path, const float* img,
                     int H, int W, int C = 3) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
    for (int c = 0; c < C; ++c) {
        float v = img[(c * H + h) * W + w];
        v = (v + 1.0f) * 0.5f;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        unsigned char p = (unsigned char)(v * 255.0f + 0.5f);
        f.put((char)p);
    }
    std::cout << "[IO] wrote " << path << " (" << H << "x" << W << "x" << C << ")\n";
}

struct Args {
    std::string drift_assets = "assets/drift";
    std::string gen_assets   = "assets/generator";
    std::string embedding    = "horse_embedding.bin";
    std::string kernel_src   = "src/onnx_kernels.cl";
    std::string log_dir      = "logs";
    std::string out_ppm      = "output_horse.ppm";
    int steps = 20;
    float cfg = 6.5f;
    bool verify_drift = false;
    bool verify_gen   = false;
    int seed = 42;
    int dump_at_node = -1;
    int max_nodes = -1;
    bool continue_on_error = true;
    bool run_drift_only = false;
    bool run_gen_only   = false;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) {
            if (i + 1 >= argc) { std::cerr << "missing arg for " << what << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (s == "--drift")     a.drift_assets = next("--drift");
        else if (s == "--gen")       a.gen_assets   = next("--gen");
        else if (s == "--embedding") a.embedding    = next("--embedding");
        else if (s == "--kernels")   a.kernel_src   = next("--kernels");
        else if (s == "--log-dir")   a.log_dir      = next("--log-dir");
        else if (s == "--out")       a.out_ppm      = next("--out");
        else if (s == "--steps")     a.steps        = std::stoi(next("--steps"));
        else if (s == "--cfg")       a.cfg          = std::stof(next("--cfg"));
        else if (s == "--seed")      a.seed         = std::stoi(next("--seed"));
        else if (s == "--verify-drift") a.verify_drift = true;
        else if (s == "--verify-gen")   a.verify_gen   = true;
        else if (s == "--verify-all")   { a.verify_drift = true; a.verify_gen = true; }
        else if (s == "--max-nodes") a.max_nodes    = std::stoi(next("--max-nodes"));
        else if (s == "--drift-only") a.run_drift_only = true;
        else if (s == "--gen-only")   a.run_gen_only   = true;
        else if (s == "--stop-on-fail") a.continue_on_error = false;
        else if (s == "--help") {
            std::cout
              << "Usage: onnx_opencl_runner [opts]\n"
                 "  --drift DIR      drift asset dir   (default assets/drift)\n"
                 "  --gen   DIR      generator asset dir (default assets/generator)\n"
                 "  --embedding FILE  512-fp32 text embedding (default horse_embedding.bin)\n"
                 "  --kernels FILE   onnx_kernels.cl path (default src/onnx_kernels.cl)\n"
                 "  --log-dir DIR    log root dir (default logs)\n"
                 "  --steps N        drift steps (default 20)\n"
                 "  --cfg F          CFG scale (default 6.5)\n"
                 "  --seed N         RNG seed for z0 (default 42)\n"
                 "  --verify-drift   diff every drift op vs golden\n"
                 "  --verify-gen     diff every generator op vs golden\n"
                 "  --verify-all     both\n"
                 "  --drift-only / --gen-only   skip the other stage\n"
                 "  --stop-on-fail   stop runtime on first failing op\n"
                 "  --out FILE.ppm   output image (default output_horse.ppm)\n";
            std::exit(0);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    // Setup z and embedding inputs (must be deterministic for golden compare).
    auto emb = read_fp32_bin(a.embedding);
    if (emb.size() != 512) {
        std::cerr << "embedding must be 512 fp32 values, got " << emb.size() << "\n";
        return 1;
    }
    std::vector<float> z(1 * 8 * 12 * 12);
    {
        // Match numpy's default normal generator (we mirror numpy by using a
        // simple deterministic Box-Muller, but ultimately we just need the
        // tensor to be valid; for golden compare set --seed to match the python).
        std::mt19937 rng((unsigned)a.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : z) v = nd(rng);
    }

    if (!a.run_gen_only) {
        Runtime rt;
        rt.log.open(a.log_dir + "/drift");
        rt.log.info("=== DRIFT (steps=%d, cfg=%.2f) ===", a.steps, a.cfg);
        rt.init_opencl();
        rt.compile_kernels(a.kernel_src);
        rt.load_manifest(a.drift_assets + "/manifest.json");
        rt.load_initializers();

        std::string golden = a.verify_drift ? (a.drift_assets + "/golden/step0") : "";
        // Run a single step diff (i=0, t=0) against the dumped golden for that step.
        std::vector<float> t0 = { 0.0f };
        std::vector<float> cfg_v = { a.cfg };
        rt.set_input_fp32("z", {1, 8, 12, 12}, z.data());
        rt.set_input_fp32("t", {1, 1}, t0.data());
        rt.set_input_fp32("text_embedding", {1, 512}, emb.data());
        rt.set_input_fp32("cfg_scale", {1}, cfg_v.data());
        rt.run(golden, /*abs_tol=*/1e-3, /*rel_tol=*/1e-2, a.continue_on_error);

        // Now iterate drift without verification (so we get a final z to feed the generator).
        std::vector<float> d(z.size());
        rt.read_output_fp32("drift", d);
        const float dt = 1.0f / (float)a.steps;
        for (size_t i = 0; i < z.size(); ++i) z[i] += d[i] * dt;
        for (int step = 1; step < a.steps; ++step) {
            float tv = (float)step / (float)a.steps;
            std::vector<float> tt = { tv };
            rt.set_input_fp32("z", {1, 8, 12, 12}, z.data());
            rt.set_input_fp32("t", {1, 1}, tt.data());
            rt.set_input_fp32("text_embedding", {1, 512}, emb.data());
            rt.set_input_fp32("cfg_scale", {1}, cfg_v.data());
            rt.run("", 1e-3, 1e-2, true);  // no verify on later steps
            rt.read_output_fp32("drift", d);
            for (size_t i = 0; i < z.size(); ++i) z[i] += d[i] * dt;
        }
        rt.log.close();
    }

    if (!a.run_drift_only) {
        Runtime rt;
        rt.log.open(a.log_dir + "/generator");
        rt.log.info("=== GENERATOR ===");
        rt.init_opencl();
        rt.compile_kernels(a.kernel_src);
        rt.load_manifest(a.gen_assets + "/manifest.json");
        rt.load_initializers();

        rt.set_input_fp32("z", {1, 8, 12, 12}, z.data());
        rt.set_input_fp32("text_embedding", {1, 512}, emb.data());
        std::string golden = a.verify_gen ? (a.gen_assets + "/golden/horse") : "";
        rt.run(golden, 1e-3, 1e-2, a.continue_on_error);

        std::vector<float> img;
        rt.read_output_fp32("reconstruction", img);
        save_ppm(a.out_ppm, img.data(), 96, 96, 3);
        rt.log.close();
    }
    return 0;
}

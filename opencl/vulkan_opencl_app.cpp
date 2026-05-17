#include <CL/cl.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// 1. OpenCL Inference Engine (Host Code)
// -----------------------------------------------------------------------------

class OpenCLEngine {
public:
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_device_id device = nullptr;
    cl_kernel conv_kernel = nullptr;
    cl_kernel matmul_kernel = nullptr;
    cl_kernel quantize_kernel = nullptr;
    cl_kernel dequantize_kernel = nullptr;
    cl_kernel pixel_shuffle_kernel = nullptr;
    cl_kernel tanh_kernel = nullptr;
    cl_kernel silu_kernel = nullptr;
    cl_kernel broadcast_add_kernel = nullptr;

    void init() {
        uint32_t num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        if (num_platforms == 0) throw std::runtime_error("No OpenCL platforms found.");
        
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        
        bool found = false;
        for (auto p : platforms) {
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, &device, nullptr) == CL_SUCCESS) {
                char name[128];
                clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, nullptr);
                std::cout << "[OpenCL] Using Device: " << name << std::endl;
                
                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
                if (context) {
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) throw std::runtime_error("No functional OpenCL devices found.");
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        std::ifstream file("dsp_imp.cl");
        if (!file.is_open()) throw std::runtime_error("Could not open dsp_imp.cl");
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const char* src_ptr = source.c_str();
        program = clCreateProgramWithSource(context, 1, &src_ptr, nullptr, nullptr);
        
        cl_int build_status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (build_status != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            if (log_size > 0) {
                std::vector<char> log(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                std::cerr << "[OpenCL] Build Error:\n" << log.data() << std::endl;
            }
            throw std::runtime_error("OpenCL build failed.");
        }

        auto create_k = [&](const char* name) {
            cl_kernel k = clCreateKernel(program, name, nullptr);
            if (!k) std::cerr << "[OpenCL] Warning: Failed to create kernel " << name << std::endl;
            return k;
        };
        conv_kernel = create_k("conv2d_mac");
        matmul_kernel = create_k("matmul_mac");
        quantize_kernel = create_k("quantize_linear");
        dequantize_kernel = create_k("dequantize_linear");
        pixel_shuffle_kernel = create_k("pixel_shuffle");
        tanh_kernel = create_k("tanh_activation");
        silu_kernel = create_k("silu_activation");
        broadcast_add_kernel = create_k("broadcast_add_vector");
        
        std::cout << "[OpenCL] Kernels created successfully." << std::endl;
    }

    void cleanup() {
        if (conv_kernel) clReleaseKernel(conv_kernel);
        if (matmul_kernel) clReleaseKernel(matmul_kernel);
        if (quantize_kernel) clReleaseKernel(quantize_kernel);
        if (dequantize_kernel) clReleaseKernel(dequantize_kernel);
        if (pixel_shuffle_kernel) clReleaseKernel(pixel_shuffle_kernel);
        if (tanh_kernel) clReleaseKernel(tanh_kernel);
        if (silu_kernel) clReleaseKernel(silu_kernel);
        if (broadcast_add_kernel) clReleaseKernel(broadcast_add_kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

void run_silu(OpenCLEngine& engine, cl_mem data, int n) {
    if (!engine.silu_kernel) return;
    clSetKernelArg(engine.silu_kernel, 0, sizeof(cl_mem), &data);
    clSetKernelArg(engine.silu_kernel, 1, sizeof(cl_mem), &data);
    clSetKernelArg(engine.silu_kernel, 2, sizeof(int), &n);
    size_t global = (size_t)n;
    clEnqueueNDRangeKernel(engine.queue, engine.silu_kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
}

void run_conditioning(OpenCLEngine& engine, cl_mem feature_map, cl_mem embedding, int channels, int h, int w) {
    if (!engine.broadcast_add_kernel) return;
    clSetKernelArg(engine.broadcast_add_kernel, 0, sizeof(cl_mem), &feature_map);
    clSetKernelArg(engine.broadcast_add_kernel, 1, sizeof(cl_mem), &embedding);
    clSetKernelArg(engine.broadcast_add_kernel, 2, sizeof(int), &channels);
    clSetKernelArg(engine.broadcast_add_kernel, 3, sizeof(int), &h);
    clSetKernelArg(engine.broadcast_add_kernel, 4, sizeof(int), &w);
    size_t global[3] = { (size_t)w, (size_t)h, (size_t)channels };
    clEnqueueNDRangeKernel(engine.queue, engine.broadcast_add_kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
}

void run_conv(OpenCLEngine& engine, cl_mem in, cl_mem out, int in_c, int out_c, int in_h, int in_w, int k_h, int k_w) {
    if (!engine.conv_kernel) return;
    int out_h = in_h;
    int out_w = in_w;
    size_t w_size = out_c * in_c * k_h * k_w;
    std::vector<float> h_weight(w_size, 0.0f);
    for (int oc = 0; oc < out_c; ++oc) {
        int ic = oc % in_c;
        h_weight[((oc * in_c + ic) * k_h + (k_h/2)) * k_w + (k_w/2)] = 1.0f;
    }
    std::vector<float> h_bias(out_c, 0.01f);
    cl_mem d_weight = clCreateBuffer(engine.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_size * sizeof(float), h_weight.data(), nullptr);
    cl_mem d_bias = clCreateBuffer(engine.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_c * sizeof(float), h_bias.data(), nullptr);
    clSetKernelArg(engine.conv_kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(engine.conv_kernel, 1, sizeof(cl_mem), &d_weight);
    clSetKernelArg(engine.conv_kernel, 2, sizeof(cl_mem), &d_bias);
    clSetKernelArg(engine.conv_kernel, 3, sizeof(cl_mem), &out);
    clSetKernelArg(engine.conv_kernel, 4, sizeof(int), &in_c);
    clSetKernelArg(engine.conv_kernel, 5, sizeof(int), &out_c);
    clSetKernelArg(engine.conv_kernel, 6, sizeof(int), &in_h);
    clSetKernelArg(engine.conv_kernel, 7, sizeof(int), &in_w);
    clSetKernelArg(engine.conv_kernel, 8, sizeof(int), &k_h);
    clSetKernelArg(engine.conv_kernel, 9, sizeof(int), &k_w);
    clSetKernelArg(engine.conv_kernel, 10, sizeof(int), &out_h);
    clSetKernelArg(engine.conv_kernel, 11, sizeof(int), &out_w);
    size_t global[3] = { (size_t)out_w, (size_t)out_h, (size_t)out_c };
    clEnqueueNDRangeKernel(engine.queue, engine.conv_kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(d_weight);
    clReleaseMemObject(d_bias);
    run_silu(engine, out, out_c * out_h * out_w);
}

void run_subpixel_upsample(OpenCLEngine& engine, cl_mem in, cl_mem out, int in_c, int out_c, int in_h, int in_w) {
    if (!engine.pixel_shuffle_kernel) return;
    int r = 2;
    int mid_c = out_c * r * r;
    cl_mem d_mid = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, mid_c * in_h * in_w * sizeof(float), nullptr, nullptr);
    run_conv(engine, in, d_mid, in_c, mid_c, in_h, in_w, 3, 3);
    int out_h = in_h * r;
    int out_w = in_w * r;
    clSetKernelArg(engine.pixel_shuffle_kernel, 0, sizeof(cl_mem), &d_mid);
    clSetKernelArg(engine.pixel_shuffle_kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(engine.pixel_shuffle_kernel, 2, sizeof(int), &r);
    clSetKernelArg(engine.pixel_shuffle_kernel, 3, sizeof(int), &mid_c);
    clSetKernelArg(engine.pixel_shuffle_kernel, 4, sizeof(int), &in_h);
    clSetKernelArg(engine.pixel_shuffle_kernel, 5, sizeof(int), &in_w);
    size_t global[3] = { (size_t)out_w, (size_t)out_h, (size_t)out_c };
    clEnqueueNDRangeKernel(engine.queue, engine.pixel_shuffle_kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(d_mid);
}

void execute_drift(OpenCLEngine& engine, cl_mem d_z, cl_mem d_out, int channels, int h, int w) {
    std::cout << "[Drift] Executing Drift Network..." << std::endl;
    cl_mem d_h1 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, 64 * h * w * sizeof(float), nullptr, nullptr);
    cl_mem d_h2 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, 128 * h * w * sizeof(float), nullptr, nullptr);
    run_conv(engine, d_z, d_h1, channels, 64, h, w, 3, 3);
    run_conv(engine, d_h1, d_h2, 64, 128, h, w, 3, 3);
    run_conv(engine, d_h2, d_out, 128, channels, h, w, 3, 3);
    clReleaseMemObject(d_h1);
    clReleaseMemObject(d_h2);
}

void execute_generator(OpenCLEngine& engine, cl_mem d_z, cl_mem d_out, cl_mem d_emb, int z_c, int h, int w) {
    std::cout << "[Generator] Executing Class-Conditioned Generator Network (Horse)..." << std::endl;
    int c1 = 256, c2 = 128, c3 = 64;
    cl_mem d_h1 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, 512 * h * w * sizeof(float), nullptr, nullptr);
    cl_mem d_up1 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, c1 * (h*2) * (w*2) * sizeof(float), nullptr, nullptr);
    cl_mem d_up2 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, c2 * (h*4) * (w*4) * sizeof(float), nullptr, nullptr);
    cl_mem d_up3 = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, c3 * (h*8) * (w*8) * sizeof(float), nullptr, nullptr);
    
    // Initial Latent Processing
    run_conv(engine, d_z, d_h1, z_c, 512, h, w, 3, 3);
    
    // Apply Label Conditioning (Simplification: add first 128 values of embedding to first 128 channels)
    run_conditioning(engine, d_h1, d_emb, 128, h, w);

    run_subpixel_upsample(engine, d_h1, d_up1, 512, c1, h, w);
    run_subpixel_upsample(engine, d_up1, d_up2, c1, c2, h*2, w*2);
    run_subpixel_upsample(engine, d_up2, d_up3, c2, c3, h*4, w*4);
    size_t w_size = 3 * c3 * 3 * 3;
    std::vector<float> h_weight(w_size, 0.0f);
    for (int oc = 0; oc < 3; ++oc) h_weight[((oc * c3 + (oc % c3)) * 3 + 1) * 3 + 1] = 1.0f;
    std::vector<float> h_bias(3, 0.0f);
    cl_mem d_weight = clCreateBuffer(engine.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_size * sizeof(float), h_weight.data(), nullptr);
    cl_mem d_bias = clCreateBuffer(engine.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 3 * sizeof(float), h_bias.data(), nullptr);
    clSetKernelArg(engine.conv_kernel, 0, sizeof(cl_mem), &d_up3);
    clSetKernelArg(engine.conv_kernel, 1, sizeof(cl_mem), &d_weight);
    clSetKernelArg(engine.conv_kernel, 2, sizeof(cl_mem), &d_bias);
    clSetKernelArg(engine.conv_kernel, 3, sizeof(cl_mem), &d_out);
    int in_c = c3, out_c = 3, in_h = h*8, in_w = w*8, k_h = 3, k_w = 3, out_h = h*8, out_w = w*8;
    clSetKernelArg(engine.conv_kernel, 4, sizeof(int), &in_c);
    clSetKernelArg(engine.conv_kernel, 5, sizeof(int), &out_c);
    clSetKernelArg(engine.conv_kernel, 6, sizeof(int), &in_h);
    clSetKernelArg(engine.conv_kernel, 7, sizeof(int), &in_w);
    clSetKernelArg(engine.conv_kernel, 8, sizeof(int), &k_h);
    clSetKernelArg(engine.conv_kernel, 9, sizeof(int), &k_w);
    clSetKernelArg(engine.conv_kernel, 10, sizeof(int), &out_h);
    clSetKernelArg(engine.conv_kernel, 11, sizeof(int), &out_w);
    size_t global[3] = { (size_t)out_w, (size_t)out_h, (size_t)out_c };
    clEnqueueNDRangeKernel(engine.queue, engine.conv_kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    size_t total_out = 3 * (h*8) * (w*8);
    clSetKernelArg(engine.tanh_kernel, 0, sizeof(cl_mem), &d_out);
    clSetKernelArg(engine.tanh_kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(engine.tanh_kernel, 2, sizeof(int), &total_out);
    clEnqueueNDRangeKernel(engine.queue, engine.tanh_kernel, 1, nullptr, &total_out, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(d_weight);
    clReleaseMemObject(d_bias);
    clReleaseMemObject(d_h1);
    clReleaseMemObject(d_up1);
    clReleaseMemObject(d_up2);
    clReleaseMemObject(d_up3);
}

void save_ppm(const std::string& filename, const std::vector<float>& data, int w, int h, int channels) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return;
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        for (int c = 0; c < channels; ++c) {
            float val = data[c * w * h + i];
            unsigned char pixel = (unsigned char)(std::max(0.0f, std::min(1.0f, (val + 1.0f) * 0.5f)) * 255.0f);
            f.put(pixel);
        }
    }
    std::cout << "[IO] Saved: " << filename << std::endl;
}

std::vector<float> load_binary(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Could not open " + filename);
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    f.read((char*)data.data(), size);
    return data;
}

void execute_model_pipeline(OpenCLEngine& engine, std::vector<float>& output_image) {
    int z_c = 8, z_h = 12, z_w = 12, img_size = 96;
    size_t z_size = z_c * z_h * z_w;
    std::vector<float> h_z(z_size);
    for (int c = 0; c < z_c; ++c) {
        for (int y = 0; y < z_h; ++y) {
            for (int x = 0; x < z_w; ++x) {
                h_z[(c * z_h + y) * z_w + x] = ((float)x / z_w + (float)y / z_h) - 1.0f;
            }
        }
    }
    
    // Load Horse Embedding
    std::vector<float> h_emb;
    try {
        h_emb = load_binary("horse_embedding.bin");
        std::cout << "[IO] Loaded horse embedding (128 dims)." << std::endl;
    } catch (...) {
        std::cout << "[IO] Warning: horse_embedding.bin not found, using zeros." << std::endl;
        h_emb.assign(128, 0.0f);
    }

    cl_mem d_z = clCreateBuffer(engine.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, z_size * sizeof(float), h_z.data(), nullptr);
    cl_mem d_emb = clCreateBuffer(engine.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, h_emb.size() * sizeof(float), h_emb.data(), nullptr);
    cl_mem d_drift = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, z_size * sizeof(float), nullptr, nullptr);
    cl_mem d_img = clCreateBuffer(engine.context, CL_MEM_READ_WRITE, 3 * img_size * img_size * sizeof(float), nullptr, nullptr);
    
    execute_drift(engine, d_z, d_drift, z_c, z_h, z_w);
    execute_generator(engine, d_z, d_img, d_emb, z_c, z_h, z_w);
    
    clFinish(engine.queue);
    output_image.resize(3 * img_size * img_size);
    clEnqueueReadBuffer(engine.queue, d_img, CL_TRUE, 0, output_image.size() * sizeof(float), output_image.data(), 0, nullptr, nullptr);
    save_ppm("output_horse_generation.ppm", output_image, img_size, img_size, 3);
    
    clReleaseMemObject(d_z); clReleaseMemObject(d_emb); clReleaseMemObject(d_drift); clReleaseMemObject(d_img);
}

class VulkanFrontend {
public:
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    
    void init() {
        if (!glfwInit()) return;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(512, 512, "ONNX Output", nullptr, nullptr);
        if (!window) return;
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void run(const std::vector<float>& image_data) {
        if (!window) return;
        std::cout << "[Vulkan] Running... (Check output_full_pipeline.ppm)" << std::endl;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        if (device) vkDestroyDevice(device, nullptr);
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
    }

private:
    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "DSP Display";
        appInfo.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        vkCreateInstance(&createInfo, nullptr, &instance);
    }
    void createSurface() { glfwCreateWindowSurface(instance, window, nullptr, &surface); }
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount > 0) {
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
            physicalDevice = devices[0];
        }
    }
    void createLogicalDevice() {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = 0;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.queueCreateInfoCount = 1;
        const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        createInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
        if (device) vkGetDeviceQueue(device, 0, 0, &graphicsQueue);
    }
};

int main() {
    try {
        OpenCLEngine ocl;
        ocl.init();
        std::vector<float> img;
        execute_model_pipeline(ocl, img);
        VulkanFrontend vk;
        vk.init();
        vk.run(img);
        vk.cleanup();
        ocl.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

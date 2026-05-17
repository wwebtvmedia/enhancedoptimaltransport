#pragma once
// Loader for the manifest.json produced by export_onnx_assets.py.

#include "tensor.hpp"
#include "../third_party/json.hpp"
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ocl {

using json = nlohmann::json;

struct AttrValue {
    // Variant of int / float / string / ints / floats / strings.
    std::string type;
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<double> floats;
    std::vector<std::string> strings;
};

struct NodeInfo {
    int index;
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, AttrValue> attrs;

    const AttrValue* attr(const std::string& k) const {
        auto it = attrs.find(k);
        return it == attrs.end() ? nullptr : &it->second;
    }
};

struct InitInfo {
    std::string name;
    std::string file;          // relative to manifest dir
    DType dtype;
    std::vector<int64_t> shape;
    size_t byte_size;
};

struct ValueInfo {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
};

struct Manifest {
    std::string root_dir;      // dir containing manifest.json
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
    std::vector<ValueInfo> value_info;
    std::vector<InitInfo> initializers;
    std::vector<NodeInfo>  nodes;

    static Manifest load(const std::string& manifest_path);
};

inline AttrValue parse_attr(const json& j) {
    AttrValue av;
    av.type = j.at("type").get<std::string>();
    const auto& v = j.at("value");
    if (av.type == "int")     av.i = v.get<int64_t>();
    else if (av.type == "float")  av.f = v.get<double>();
    else if (av.type == "string") av.s = v.get<std::string>();
    else if (av.type == "ints")   av.ints  = v.get<std::vector<int64_t>>();
    else if (av.type == "floats") av.floats = v.get<std::vector<double>>();
    else if (av.type == "strings") av.strings = v.get<std::vector<std::string>>();
    return av;
}

inline Manifest Manifest::load(const std::string& manifest_path) {
    std::ifstream f(manifest_path);
    if (!f.is_open()) throw std::runtime_error("manifest not found: " + manifest_path);
    json j;
    f >> j;

    Manifest m;
    // root_dir = dirname(manifest_path)
    {
        size_t pos = manifest_path.find_last_of("/\\");
        m.root_dir = (pos == std::string::npos) ? "." : manifest_path.substr(0, pos);
    }

    auto parse_vi = [](const json& jvi) {
        ValueInfo vi;
        vi.name = jvi.at("name").get<std::string>();
        vi.dtype = dtype_from_str(jvi.at("dtype").get<std::string>());
        for (auto& d : jvi.at("shape")) vi.shape.push_back(d.get<int64_t>());
        return vi;
    };
    for (auto& jv : j.at("inputs"))     m.inputs.push_back(parse_vi(jv));
    for (auto& jv : j.at("outputs"))    m.outputs.push_back(parse_vi(jv));
    for (auto& jv : j.at("value_info")) m.value_info.push_back(parse_vi(jv));

    for (auto& ji : j.at("initializers")) {
        InitInfo ii;
        ii.name = ji.at("name").get<std::string>();
        ii.file = ji.at("file").get<std::string>();
        ii.dtype = dtype_from_str(ji.at("dtype").get<std::string>());
        for (auto& d : ji.at("shape")) ii.shape.push_back(d.get<int64_t>());
        ii.byte_size = ji.at("byte_size").get<size_t>();
        m.initializers.push_back(std::move(ii));
    }

    for (auto& jn : j.at("nodes")) {
        NodeInfo n;
        n.index = jn.at("index").get<int>();
        n.name = jn.at("name").get<std::string>();
        n.op_type = jn.at("op_type").get<std::string>();
        for (auto& x : jn.at("inputs"))  n.inputs.push_back(x.get<std::string>());
        for (auto& x : jn.at("outputs")) n.outputs.push_back(x.get<std::string>());
        if (jn.contains("attrs")) {
            for (auto it = jn["attrs"].begin(); it != jn["attrs"].end(); ++it) {
                n.attrs.emplace(it.key(), parse_attr(it.value()));
            }
        }
        m.nodes.push_back(std::move(n));
    }
    return m;
}

} // namespace ocl

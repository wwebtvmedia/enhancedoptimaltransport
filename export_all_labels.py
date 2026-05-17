import json
import struct
import os

def export_embeddings(json_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    for label, emb in data.items():
        filename = os.path.join(output_dir, f"label_{label}.bin")
        with open(filename, 'wb') as f_out:
            f_out.write(struct.pack('f'*len(emb), *emb))
        print(f"Exported: {filename}")

if __name__ == "__main__":
    export_embeddings("enhanced_label_sb/onnx/label_embeddings.json", "opencl/assets")

import numpy as np
from PIL import Image
import sys
import os

def convert_to_png(input_file, output_file, width=128, height=128):
    try:
        # Read the raw text data from simulation dump
        with open(input_file, 'r') as f:
            data = [int(line.strip()) for line in f if line.strip()]
        
        if len(data) < width * height:
            print(f"Error: Data size ({len(data)}) is less than expected ({width}x{height})")
            return

        # Reshape to image array
        img_array = np.array(data[:width*height], dtype=np.uint32)
        
        # Normalize to 0-255 for grayscale PNG display (clipping any overflow)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_array = img_array.reshape((height, width))
        
        # Save as PNG
        img = Image.fromarray(img_array)
        img.save(output_file)
        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    infile = "output_image.txt"
    outfile = "output_image.png"
    
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
        
    if os.path.exists(infile):
        convert_to_png(infile, outfile)
    else:
        print(f"Input file {infile} not found. Run the VHDL simulation first.")

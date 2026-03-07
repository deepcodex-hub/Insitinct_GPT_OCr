import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import sys
from scripts.generate_debug_session import process_and_save_debug

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    out_dir = "examples/debug_session/single_test"
    print(f"Testing image: {img_path}")
    process_and_save_debug(img_path, out_dir, "test_screenshot")
    print(f"Results saved to {out_dir}")

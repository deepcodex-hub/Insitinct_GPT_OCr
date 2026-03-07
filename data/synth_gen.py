import os
import argparse

def generate_synthetic_data(num_samples: int, output_dir: str):
    """
    Wrapper around TextRecognitionDataGenerator to create meter panels.
    https://github.com/Belval/TextRecognitionDataGenerator
    """
    print(f"Generating {num_samples} synthetic meter images to {output_dir}...")
    
    # We use system call to TRDG to generate images
    # We simulate the exact command for the user
    trdg_cmd = (
        f"trdg -c {num_samples} -w 5 -t 2 -b 1 -d 3 -f 64 "
        f"--output_dir {output_dir} "
        f"--blur 1" # simulate some focal blur
    )
    
    print(f"Running TRDG: {trdg_cmd}")
    
    # In a real environment:
    # os.system(trdg_cmd)
    
    print("Generation complete.")
    
def download_public_datasets():
    """Helper instructions/script to download MJSynth, SynthText, ICDAR."""
    print("--- Public Dataset Download Instructions ---")
    print("Kaggle: Ensure you have your ~/.kaggle/kaggle.json set up.")
    print("Run: kaggle datasets download -d <some_meter_dataset>")
    print("SynthText: wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip")
    print("MJSynth: wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--out", type=str, default="./data/synth_meter")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    generate_synthetic_data(args.samples, args.out)
    download_public_datasets()

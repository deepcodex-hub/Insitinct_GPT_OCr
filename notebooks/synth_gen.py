import os
import random
import argparse
from trdg.generators import GeneratorFromStrings

def generate_synth_meters(count, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Strings simulating meter readings
    strings = [str(random.randint(10000, 9999999)) for _ in range(count)]
    
    generator = GeneratorFromStrings(
        strings,
        blur=1,
        random_blur=True,
        distorsion_type=3, # Random distortion
        alignment=1, # Center
        background_type=2, # Quizzical
        text_color='#000000',
        font_size=60
    )

    for i, (img, lbl) in enumerate(generator):
        if i >= count: break
        img_path = os.path.join(output_dir, f"meter_{i}.jpg")
        img.save(img_path)
        with open(os.path.join(output_dir, "labels.csv"), "a") as f:
            f.write(f"{img_path},{lbl}\n")
            
    print(f"Generated {count} synthetic meter images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", default="data/synthetic")
    args = parser.parse_args()
    
    generate_synth_meters(args.count, args.output)

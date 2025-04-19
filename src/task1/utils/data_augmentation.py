import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import glob
import io

def augment_dataset(input_dir, output_dir, morph_exec="./imagemorph", augment_per_image=1, alpha="0.9", kernel_size="9"):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(".pgm"):
                continue

            input_image_path = os.path.join(class_path, filename)
            base_name = os.path.splitext(filename)[0]

            try:
                img = Image.open(input_image_path).convert("RGB")
            except Exception as e:
                print(f"Error reading the image: {input_image_path} — {e}")
                continue

            # Save the image in PPM format to a buffer
            ppm_buffer = io.BytesIO()
            img.save(ppm_buffer, format="PPM")
            ppm_buffer.seek(0)

            for i in range(augment_per_image):
                output_filename = f"{base_name}_morph{i+1}.pgm"
                output_image_path = os.path.join(output_class_path, output_filename)

                with open(output_image_path, "wb") as fout:
                    subprocess.run(
                        [morph_exec, alpha, kernel_size],
                        input=ppm_buffer.getvalue(),
                        stdout=fout,
                        stderr=subprocess.DEVNULL
                    )

                print(f"Saved: {output_image_path}")

                # Bring back the buffer to the start for the next iteration
                ppm_buffer.seek(0)
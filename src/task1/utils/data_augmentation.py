import os
import io
import subprocess
from PIL import Image

import os
import io
import subprocess
from PIL import Image

def augment_dataset(image_paths, labels, output_dir, morph_exec="./imagemorph",
                    augment_per_image=1, alpha="0.9", kernel_size="9"):

    os.makedirs(output_dir, exist_ok=True)
    augmented_paths, augmented_labels = [], []

    for idx, input_image_path in enumerate(image_paths):
        label = labels[idx] if labels else ""
        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_image_path))[0]

        img = Image.open(input_image_path).convert("RGB")
        ppm_buffer = io.BytesIO()
        img.save(ppm_buffer, format="PPM")
        ppm_buffer.seek(0)

        for i in range(augment_per_image):
            output_filename = f"{base_name}_morph{i+1}.pgm"
            output_image_path = os.path.join(label_output_dir, output_filename)

            with open(output_image_path, "wb") as fout:
                subprocess.run(
                    [morph_exec, alpha, kernel_size],
                    input=ppm_buffer.getvalue(),
                    stdout=fout,
                    stderr=subprocess.DEVNULL
                )

            augmented_paths.append(output_image_path)
            augmented_labels.append(label)

            ppm_buffer.seek(0)

    return augmented_paths, augmented_labels

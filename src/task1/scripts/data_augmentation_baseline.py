from src.task1.utils.data_augmentation import *

# Configuration
input_dir = "monkbrill"
output_dir = "augmented"
morph_exec = "./imagemorph"
augment_per_image = 1
alpha = "0.9"
kernel_size = "9"

augment_dataset(
    input_dir=input_dir,
    output_dir=output_dir,
    morph_exec=morph_exec,
    augment_per_image=augment_per_image,
    alpha=alpha,
    kernel_size=kernel_size
)

# output_class_path = "augmented/Resh"
# base_name = "navis-QIrug-Qumran_extr09_0004-line-034-y1=2948-y2=3086-zone-HUMAN-x=0550-y=0066-w=0038-h=0039-ybas=0076-nink=489-segm=COCOS5cocos"

# show_augmented_variants(output_class_path, base_name)
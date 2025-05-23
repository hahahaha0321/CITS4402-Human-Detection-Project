import os
from PIL import Image

def add_png_extension(folder_path):
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and not filename.lower().endswith('.png'):
            new_path = full_path + '.png'
            os.rename(full_path, new_path)
            print(f"Renamed: {filename} → {os.path.basename(new_path)}")

def crop_to_1_2_aspect(img):
    width, height = img.size
    target_ratio = 0.5  # 1:2 aspect ratio (H:W)
    current_ratio = height / width

    if current_ratio > target_ratio:
        # Image is too tall — crop height
        new_height = int(width * target_ratio)
        top = (height - new_height) // 2
        return img.crop((0, top, width, top + new_height))
    else:
        # Image is too wide — crop width
        new_width = int(height / target_ratio)
        left = (width - new_width) // 2
        return img.crop((left, 0, left + new_width, height))


import os
import shutil
import random

image_dir = 'dataset/images'
label_dir = 'dataset/labels'
split_dir = 'dataset/split'
os.makedirs(split_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
random.shuffle(images)
train_split = int(0.7 * len(images))  # 70% train
val_split = int(0.85 * len(images))   # 15% val, 15% test

for split, start, end in [
    ('train', 0, train_split),
    ('val', train_split, val_split),
    ('test', val_split, len(images))
]:
    os.makedirs(f'{split_dir}/images/{split}', exist_ok=True)
    os.makedirs(f'{split_dir}/labels/{split}', exist_ok=True)
    for img in images[start:end]:
        shutil.copy(f'{image_dir}/{img}', f'{split_dir}/images/{split}/{img}')
        label = img.replace('.png', '.txt')
        if os.path.exists(f'{label_dir}/{label}'):
            shutil.copy(f'{label_dir}/{label}', f'{split_dir}/labels/{split}/{label}')
        else:
            open(f'{split_dir}/labels/{split}/{label}', 'w').close()
print('Dataset split completed')
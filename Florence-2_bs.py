import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

# 任务和模型名称
task = "<MORE_DETAILED_CAPTION>"
model_name = r"/mnt/e/public_model/florence2-large-ft-gufeng_v3.1"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和处理器到指定设备
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# 批处理大小
batch_size = 28

# 获取目录路径
directory = Path(r"/mnt/e/LFH/lfhnoc")

# 支持的图片格式
patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.webp']
patterns += [p.upper() for p in patterns]  # 添加大写格式
filenames = [fn for pattern in patterns for fn in directory.glob(pattern)]

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return str(self.file_paths[idx])  # 将 Path 对象转换为字符串

# 自定义 collate_fn
def collate_fn(batch):
    return batch

# 数据加载器
data_loader = DataLoader(ImageDataset(filenames), batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)

@torch.inference_mode()
def process_images(images):
    inputs = processor(text=[task] * len(images), images=images, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
    )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts

# 处理每个批次的图片
for batch in tqdm(data_loader):
    images = []
    to_process = []
    for filename in batch:
        output_filename = Path(filename).with_suffix(".txt")
        if output_filename.exists():
            continue  # 如果输出文件已经存在，则跳过

        with Image.open(filename) as img:
            img = img.convert("RGB")
            images.append(img)
            to_process.append(filename)

    if not images:
        continue  # 如果没有图片需要处理，跳过本次循环

    generated_texts = process_images(images)

    # 将生成的文字保存到对应的文件
    for i, caption in enumerate(generated_texts):
        filename = Path(to_process[i])
        with open(filename.with_suffix(".txt"), "w") as text_file:
            text_file.write(caption)

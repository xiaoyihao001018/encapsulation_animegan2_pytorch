import sys
import torch
from PIL import Image
from model import Generator
from torchvision.transforms.functional import to_tensor, to_pil_image
import argparse

def process_image(input_path, output_path, checkpoint_path, device="cpu"):
    # 加载模型
    net = Generator()
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.to(device).eval()
    
    # 加载图片
    img = Image.open(input_path).convert("RGB")
    
    # 处理图片
    with torch.no_grad():
        image = to_tensor(img).unsqueeze(0) * 2 - 1
        out = net(image.to(device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)
    
    # 保存结果
    out.save(output_path)
    print(f"处理完成! 结果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='动漫风格化处理程序')
    parser.add_argument('--input', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output', type=str, required=True, help='输出图片路径')
    parser.add_argument('--checkpoint', type=str, default='pytorch_generator_Hayao.pt', help='模型文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='使用设备 (cpu/cuda)')
    
    args = parser.parse_args()
    process_image(args.input, args.output, args.checkpoint, args.device)

if __name__ == "__main__":
    main() 
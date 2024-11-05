# 导入必要的库
from fastapi import FastAPI, UploadFile, File, HTTPException  # FastAPI框架相关
from fastapi.responses import FileResponse     # 用于返回文件
import torch                                   # PyTorch深度学习框架
from PIL import Image                         # 图像处理库
import io                                     # 用于处理二进制流
import os                                     # 操作系统接口
import sys                                    # 系统相关功能
import time                                   # 时间处理
from rich.console import Console              # 美化控制台输出
from rich.progress import Progress, SpinnerColumn, TextColumn  # 进度显示组件
import uvicorn                                # ASGI服务器
from torchvision.transforms.functional import to_tensor, to_pil_image  # 添加这行导入

# 初始化控制台对象，用于美化输出
console = Console()

def show_banner():
    """显示启动横幅"""
    console.print("""
[bold cyan]AnimeGAN v2 服务[/bold cyan]
[green]启动中...[/green]
""")

def init_model():
    """
    初始化模型
    返回: 初始化好的模型实例
    """
    # 创建进度显示器
    with Progress(
        SpinnerColumn(),                      # 添加旋转动画
        TextColumn("[progress.description]{task.description}"),  # 显示任务描述
        transient=True,                       # 完成后清除进度条
    ) as progress:
        try:
            # 第一阶段：初始化模型
            task1 = progress.add_task("[cyan]正在初始化模型...", total=None)
            device = "cpu"                    # 设置运行设备为CPU
            # 从torch hub加载预训练模型
            model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main", 
                "generator", 
                device=device,
                progress=True,
                verbose=False
            )
            progress.update(task1, completed=True)
            
            # 第二阶段：准备模型
            task2 = progress.add_task("[cyan]正在准备模型...", total=None)
            model.eval()                      # 设置为评估模式
            progress.update(task2, completed=True)
            
            console.print("[bold green]✓[/bold green] 模型加载完成！")
            return model
            
        except Exception as e:
            console.print(f"[bold red]✗ 错误：{str(e)}[/bold red]")
            sys.exit(1)

# 创建FastAPI应用实例
app = FastAPI(title="AnimeGAN v2 服务")

# 显示启动横幅
show_banner()

# 初始化模型
device = "cpu"
model = init_model()

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    """
    图片转换接口
    参数:
        file: 上传的图片文件
    返回:
        转换后的图片文件
    """
    try:
        # 读取上传的图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 处理图片
        with torch.no_grad():                 # 禁用梯度计算
            image = to_tensor(image).unsqueeze(0) * 2 - 1  # 预处理图片
            out = model(image.to(device)).cpu()[0]         # 模型推理
            out = (out * 0.5 + 0.5).clip(0, 1)            # 后处理
            out = to_pil_image(out)                        # 转换为PIL图像
        
        # 保存结果
        output_path = "temp_output.png"
        out.save(output_path)
        
        # 返回结果图片
        return FileResponse(
            output_path,
            media_type="image/png",
        )
    except Exception as e:
        console.print(f"[bold red]✗ 错误：{str(e)}[/bold red]")
        return {"error": "处理图片时发生错误"}

# 主程序入口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600)  # 启动服务器 
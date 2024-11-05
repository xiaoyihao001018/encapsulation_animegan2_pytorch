from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from PIL import Image
import io
import os
from model import Generator
from torchvision.transforms.functional import to_tensor, to_pil_image
import uvicorn

app = FastAPI()

# 初始化模型
device = "cpu"
# 使用torch.hub加载模型
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device)
model.eval()

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    # 读取上传的图片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 处理图片
    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = model(image.to(device)).cpu()[0]
        out = (out * 0.5 + 0.5).clip(0, 1)
        out = to_pil_image(out)
    
    # 保存结果
    output_path = "temp_output.png"
    out.save(output_path)
    
    # 返回结果图片
    return FileResponse(output_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600)
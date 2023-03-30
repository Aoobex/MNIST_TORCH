# name:Aoobex
import torch
import torchvision
from PIL import Image

from module import aoobex_module

module = torch.load("train_10",map_location=torch.device("cpu"))
img = Image.open("THREE.png")
img = img.convert("1")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)),
                                      torchvision.transforms.ToTensor()])

img = transform(img)
img = torch.reshape(img,(-1,1,28,28))
output = module(img)
print(output.argmax(1))
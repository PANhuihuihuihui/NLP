import timm 
import torch

model = timm.create_model('resnet34')
x     = torch.randn(1, 3, 224, 224)
print(model(x).shape)

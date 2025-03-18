import torch
from torch import nn
import torchvision

# create func
def create_effnetb2(num_classes=3):
  weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms=weights.transforms()
  model=torchvision.models.efficientnet_b2(weights=weights)
  for param in model.parameters():
    param.requires_grad=False

    model.classifier=nn.Sequential(
    nn.Dropout(p=0.3,inplace=True),
    nn.Linear(in_features=1408,out_features=num_classes,bias=True)
  )
  return model,transforms

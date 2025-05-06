import logging
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from rich.logging import RichHandler

import pytorch2timeloop

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "../converted")

# sys run rm -rf converted
if os.path.exists(SAVE_DIR):
    print(f"Removing {SAVE_DIR}")
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "arch"), exist_ok=True)

########### Converting AlexNet ###########

net = models.alexnet()
with open(os.path.join(SAVE_DIR, "arch/alexnet.txt"), "w") as f:
    f.write(str(net))
    f.write("\n")

pytorch2timeloop.convert_model(
    model=net,
    input_size=(3, 224, 224),
    batch_size=1,
    convert_fc=True,
    model_name='alexnet',
    save_dir=SAVE_DIR,
    fuse=False,
    exception_module_names=["flatten", "avgpool", "maxpool"]
)

# ########### Converting ResNet ###########

net = models.resnet18()

with open(os.path.join(SAVE_DIR, "arch/resnet18.txt"), "w") as f:
    f.write(str(net))
    f.write("\n")

pytorch2timeloop.convert_model(
    model=net,
    input_size=(3, 224, 224),
    batch_size=1,
    convert_fc=True,
    model_name="resnet18",
    save_dir=SAVE_DIR,
    fuse=False,
    exception_module_names=["flatten", "avgpool", "maxpool", "add"],
)

###### Converting ConvNeXt ###########
net = models.convnext_tiny()

with open(os.path.join(SAVE_DIR, "arch/convnext.txt"), "w") as f:
    f.write(str(net))
    f.write("\n")

pytorch2timeloop.convert_model(
    model=net,
    input_size=(3, 224, 224),
    batch_size=1,
    convert_fc=True,
    model_name="convnext",
    save_dir=SAVE_DIR,
    fuse=False,
    ignored_func=[
        F.layer_norm,
        torch.permute,
        torchvision.ops.stochastic_depth,
    ],
    ignored_modules=(nn.LayerNorm, nn.Flatten, nn.AdaptiveAvgPool2d),
    exception_module_names=["add", "mul"],
)

############## Converting ViT ###########
# from transformers import ViTConfig, ViTModel

# # Load a pre-trained ViT model
# model = ViTModel.from_pretrained('google/vit-base-patch16-224')
# model.eval()  # Set to evaluation mode

# # Print model architecture
# print(model)

# pytorch2timeloop.convert_model(
#     model=model,
#     input_size=(3, 224, 224),
#     batch_size=1,
#     convert_fc=True,
#     model_name='vit',
#     save_dir="converted",
#     fuse=False,
#     exception_module_names=["flatten", "avgpool", "maxpool"]
# )

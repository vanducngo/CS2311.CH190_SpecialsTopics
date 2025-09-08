import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch
import os

from RoTTA.core.utils.constants import DEVICE

def get_pretrained_model(cfg):
    model_path = './ckpt/MobileNet_Sep06_22h05.pth'
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def get_model_chexpert(cfg):
    return get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)

def get_model(cfg, feature_extract=False, useWeight=True, numclasses=5):
    model = None
    arch = cfg.MODEL.ARCH.lower()
    
    print(f">>> Loading model: {arch} | useWeight: {useWeight} | num_classes: {numclasses}")

    if arch == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
        model = mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Model architecture {arch} not supported.")

    if hasattr(model, 'classifier'):
        # DenseNet (classifier is a Linear)
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, numclasses)
            )
        # MobileNet (classifier is a Sequential)
        elif isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512), # Lớp ẩn mới
                nn.ReLU(),
                nn.Linear(512, numclasses)
            )
        else:
            raise TypeError(f"Unsupported classifier type: {type(model.classifier)}")
    else:
        raise AttributeError("Model does not have 'fc' or 'classifier' attribute.")


    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {numclasses} classes.")
    return model
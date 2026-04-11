import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer:
    """Mixin class for LoRA layers with common properties."""
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        self.scaling = lora_alpha / r if r > 0 else 1.0

class LinearLoRA(nn.Module, LoRALayer):
    """Low-Rank Adaptation for Linear layers."""
    def __init__(self, base_layer: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout)
        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        
        # Kaiming initialization for A and Zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

class Conv2dLoRA(nn.Module, LoRALayer):
    """Low-Rank Adaptation for Conv2d layers using 1x1 convolutions."""
    def __init__(self, base_layer: nn.Conv2d, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout)
        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
            
        # We use 1x1 convolutions for the low-rank path
        self.lora_A = nn.Conv2d(base_layer.in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, base_layer.out_channels, kernel_size=1, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

def apply_lora(model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["Linear", "Conv2d"]):
    """
    Recursively replaces target modules with their LoRA counterparts.
    Targets specific layer types based on the target_modules list.
    """
    num_wrapped = 0
    for name, module in model.named_children():
        # Check if module is a target
        is_target = any(target in str(type(module)) for target in target_modules)
        
        if is_target:
            if isinstance(module, nn.Linear):
                # Don't wrap already wrapped layers
                if not isinstance(module, LinearLoRA):
                    setattr(model, name, LinearLoRA(module, r, lora_alpha, lora_dropout))
                    num_wrapped += 1
            elif isinstance(module, nn.Conv2d):
                if not isinstance(module, Conv2dLoRA):
                    setattr(model, name, Conv2dLoRA(module, r, lora_alpha, lora_dropout))
                    num_wrapped += 1
        
        # Recurse into children
        num_wrapped += apply_lora(module, r, lora_alpha, lora_dropout, target_modules)
        
    return num_wrapped

def get_lora_params(model):
    """Returns only the parameters created for LoRA."""
    return [p for n, p in model.named_parameters() if "lora_" in n]

def count_lora_params(model):
    """Counts total and trainable parameters to show the reduction."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

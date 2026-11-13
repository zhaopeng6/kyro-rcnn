"""The class-specific crop classifier used after class-agnostic segmentation."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16


# AutoKary uses 23 and 24 for the sex chromosomes.
CLASS_NAMES = tuple([str(number) for number in range(1, 23)] + ["X", "Y"])
_RAW_TO_INDEX = {str(number): number - 1 for number in range(1, 23)} | {"23": 22, "24": 23, "X": 22, "Y": 23}
LORA_DEFAULT_RANK = 8
HF_VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"


def label_index(raw_label: str | int | None) -> int | None:
    """Convert an AutoKary label to a zero-based classifier target."""
    return _RAW_TO_INDEX.get(str(raw_label).strip().upper()) if raw_label is not None else None


def class_name(index: int) -> str:
    return CLASS_NAMES[int(index)]


def _resolve_vit_lora_targets(model: nn.Module) -> list[str]:
    """Support both older query/value and newer q_proj/v_proj HF ViT names."""
    linear_leaf_names = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    for candidates in (("query", "value"), ("q_proj", "v_proj")):
        if set(candidates).issubset(linear_leaf_names): return list(candidates)
    available = sorted(linear_leaf_names)
    raise RuntimeError(f"Could not find ViT attention projections for LoRA. Expected query/value or q_proj/v_proj; linear module names found: {available}")


def _disable_incompatible_torchao() -> str | None:
    """Avoid a Kaggle PEFT failure caused by its stale optional torchao build.

    PEFT checks torchao before determining that the target is an ordinary
    ``nn.Linear``. Some Kaggle images bundle a torchao release that PEFT rejects
    outright, even though no torchao quantized layer is in this model. In that
    case, disable just the optional dispatcher and use PEFT's normal PyTorch
    LoRA implementation.
    """
    try:
        from peft.tuners.lora import torchao as peft_torchao
        peft_torchao.is_torchao_available()
    except ModuleNotFoundError:
        return None
    except ImportError as error:
        peft_torchao.is_torchao_available = lambda: False
        return str(error)
    return None


def _build_lora_vit(pretrained: bool, num_classes: int, rank: int, alpha: int, dropout: float) -> nn.Module:
    """Build a Hugging Face ViT with PEFT adapters on query/value attention."""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import ViTConfig, ViTForImageClassification
    except ModuleNotFoundError as exc:
        raise RuntimeError("LoRA ViT requires `transformers` and `peft`. Install project requirements, e.g. `python -m pip install -r requirements.txt`.") from exc
    label2id = {name: index for index, name in enumerate(CLASS_NAMES)}
    base = (ViTForImageClassification.from_pretrained(HF_VIT_CHECKPOINT, num_labels=num_classes, id2label={index: name for name, index in label2id.items()}, label2id=label2id, ignore_mismatched_sizes=True)
            if pretrained else ViTForImageClassification(ViTConfig(num_labels=num_classes, id2label={index: name for name, index in label2id.items()}, label2id=label2id)))
    targets = _resolve_vit_lora_targets(base)
    torchao_warning = _disable_incompatible_torchao()
    # PEFT has no IMAGE_CLASSIFICATION TaskType. Leaving task_type unset uses
    # generic adapter injection, while modules_to_save keeps the newly created
    # 24-class classifier trainable and serialised with the adapter.
    lora = LoraConfig(r=int(rank), lora_alpha=int(alpha), lora_dropout=float(dropout), target_modules=targets, modules_to_save=["classifier"], bias="none")
    model = get_peft_model(base, lora)
    model._chromosome_lora_targets = targets
    model._chromosome_torchao_warning = torchao_warning
    return model


def build_chromosome_classifier(architecture: str = "resnet18", pretrained: bool = True, num_classes: int = len(CLASS_NAMES), finetuning_mode: str = "full", lora_rank: int = LORA_DEFAULT_RANK, lora_alpha: int | None = None, lora_dropout: float = 0.0) -> nn.Module:
    """Build a 24-way ResNet-18 or ViT-B/16 classifier.

    For either architecture, pretrained ImageNet features are retained and only
    the final prediction layer is replaced before the trainer fine-tunes all
    parameters.
    """
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "vit_b_16" and finetuning_mode == "lora":
        return _build_lora_vit(pretrained, num_classes, lora_rank, lora_alpha or lora_rank, lora_dropout)
    if architecture == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier architecture/mode: {architecture!r}/{finetuning_mode!r}. LoRA is available only for 'vit_b_16'.")


def classifier_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Normalize torchvision and Hugging Face/PEFT model forward outputs."""
    if hasattr(model, "peft_config"):
        return model(pixel_values=images).logits
    return model(images)


def load_classifier(checkpoint: str | Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mode = payload.get("finetuning_mode", "full")
    model = build_chromosome_classifier(payload.get("architecture", "resnet18"), pretrained=(mode == "lora"), num_classes=int(payload.get("num_classes", len(CLASS_NAMES))), finetuning_mode=mode, lora_rank=int(payload.get("lora_rank", LORA_DEFAULT_RANK)), lora_alpha=payload.get("lora_alpha"), lora_dropout=float(payload.get("lora_dropout", 0.0)))
    if mode == "lora":
        try:
            from peft import set_peft_model_state_dict
        except ModuleNotFoundError as exc:
            raise RuntimeError("Loading a LoRA classifier requires `peft`. Install project requirements.") from exc
        set_peft_model_state_dict(model, payload["adapter_state"])
    else:
        model.load_state_dict(payload["model"] if "model" in payload else payload)
    return model.to(device).eval()

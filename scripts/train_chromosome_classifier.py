#!/usr/bin/env python3
"""Train the 24-way chromosome-type classifier on AutoKary mask-isolated crops."""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chromosome_segmentation"))
from src.karyogram.classifier import CLASS_NAMES, LORA_DEFAULT_RANK, build_chromosome_classifier, classifier_logits
from src.karyogram.dataset import AutoKaryCropDataset
from src.karyogram.device import resolve_device
from src.utils.tensorboard import TrainingTensorBoard
from src.utils.config import dataset_path


def configure_trainable_layers(model, architecture: str, freeze_backbone: bool) -> int:
    """Freeze/unfreeze feature layers while retaining the task-specific head."""
    for parameter in model.parameters(): parameter.requires_grad = not freeze_backbone
    if freeze_backbone:
        head = model.fc if architecture == "resnet18" else model.heads
        for parameter in head.parameters(): parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _run_epoch(model, loader, criterion, optimizer, device, collect_per_class=False):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    class_correct = torch.zeros(len(CLASS_NAMES), dtype=torch.long) if collect_per_class else None
    class_total = torch.zeros(len(CLASS_NAMES), dtype=torch.long) if collect_per_class else None
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = classifier_logits(model, images); loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            predicted = logits.argmax(1)
            total_correct += int((predicted == labels).sum())
            if collect_per_class:
                for label in labels.unique():
                    label_index = int(label)
                    selected = labels == label
                    class_total[label_index] += selected.sum().cpu()
                    class_correct[label_index] += (predicted[selected] == label).sum().cpu()
            total_items += len(labels)
    result = {"loss": total_loss / max(1, total_items), "accuracy": total_correct / max(1, total_items), "samples": total_items}
    if collect_per_class:
        result["per_class_accuracy"] = {CLASS_NAMES[index]: (int(class_correct[index]) / int(class_total[index]) if class_total[index] else None) for index in range(len(CLASS_NAMES))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp_karyogram.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a classifier checkpoint (usually output_dir/last.pt).")
    parser.add_argument("--check-data", action="store_true", help="Print resolved dataset paths/counts and exit; useful before a Kaggle run.")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    options = cfg["classifier_training"]
    architecture = str(options.get("architecture", "resnet18"))
    finetuning_mode = str(options.get("finetuning_mode", "full"))
    lora_options = options.get("lora", {})
    lora_rank = int(lora_options.get("rank", LORA_DEFAULT_RANK))
    lora_alpha = int(lora_options.get("alpha", lora_rank))
    lora_dropout = float(lora_options.get("dropout", 0.0))
    if finetuning_mode not in ("full", "lora"): raise ValueError("classifier_training.finetuning_mode must be 'full' or 'lora'")
    if finetuning_mode == "lora" and architecture != "vit_b_16": raise ValueError("LoRA fine-tuning currently requires classifier_training.architecture: vit_b_16")
    if finetuning_mode == "lora" and lora_rank < 1: raise ValueError("classifier_training.lora.rank must be positive")
    freeze_backbone_epochs = int(options.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs < 0: raise ValueError("classifier_training.freeze_backbone_epochs must be zero or positive")
    if architecture == "vit_b_16" and int(options.get("image_size", 224)) != 224:
        raise ValueError("torchvision vit_b_16 requires classifier_training.image_size: 224")
    train_root = dataset_path(cfg, cfg["datasets"]["autokary"]["root"])
    validation_root = dataset_path(cfg, cfg["datasets"]["autokary"]["validation_root"])
    for split, root in (("training", train_root), ("validation", validation_root)):
        if not root.is_dir():
            raise FileNotFoundError(f"{split.capitalize()} AutoKary directory does not exist: {root}\n"
                                f"paths.data_root={cfg.get('paths', {}).get('data_root', '.')} and the configured datasets.autokary path were combined. "
                                "On Kaggle, use `find /kaggle/input -type d -name train_labelme` to locate the attached dataset, then update paths.data_root and/or datasets.autokary.root.")
    train_data = AutoKaryCropDataset(train_root, options.get("crop_padding", 12), options.get("image_size", 224), options.get("normalization", "imagenet"))
    validation_data = AutoKaryCropDataset(validation_root, options.get("crop_padding", 12), options.get("image_size", 224), options.get("normalization", "imagenet"))
    print(json.dumps({"dataset_paths": {"train": str(train_root), "validation": str(validation_root)}, "labelled_chromosome_crops": {"train": len(train_data), "validation": len(validation_data)}}))
    if not len(train_data):
        json_count = sum(1 for _ in train_root.rglob("*.json"))
        raise RuntimeError(f"No valid AutoKary chromosome crops found under {train_root} ({json_count} JSON files found). "
                           "The classifier requires LabelMe annotation JSON files with polygon shapes labelled 1-24. Check the Kaggle directory layout and datasets.autokary.root.")
    if args.check_data: return
    device = resolve_device(args.device or options.get("device", "cpu"))
    loader_args = dict(batch_size=int(options.get("batch_size", 32)), num_workers=int(options.get("num_workers", 0)), pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    model = build_chromosome_classifier(architecture, bool(options.get("pretrained", True)), finetuning_mode=finetuning_mode, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout).to(device)
    if finetuning_mode == "lora": print(json.dumps({"lora_target_modules": getattr(model, "_chromosome_lora_targets", "resolved by PEFT"), "lora_rank": lora_rank, "lora_alpha": lora_alpha}))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(options.get("learning_rate", 3e-4)), weight_decay=float(options.get("weight_decay", 1e-4)))
    criterion = nn.CrossEntropyLoss()
    output = Path(options.get("output_dir", "outputs/chromosome_classifier")); output.mkdir(parents=True, exist_ok=True)
    tensorboard_options = options.get("tensorboard", {})
    tensorboard = TrainingTensorBoard(tensorboard_options.get("log_dir", output / "tensorboard"), tensorboard_options.get("enabled", True))
    requested_epochs = int(args.epochs or options.get("epochs", 20))
    history, start_epoch = [], 1
    configured_resume = options.get("resume_checkpoint")
    resume_path = Path(args.resume or configured_resume) if (args.resume or configured_resume) else None
    if resume_path:
        if not resume_path.exists(): raise FileNotFoundError(f"Classifier resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_architecture = checkpoint.get("architecture", "resnet18")
        checkpoint_mode = checkpoint.get("finetuning_mode", "full")
        if checkpoint_architecture != architecture or checkpoint_mode != finetuning_mode:
            raise ValueError(f"Checkpoint is {checkpoint_architecture!r}/{checkpoint_mode!r}, but the config requests {architecture!r}/{finetuning_mode!r}. Use a matching config or start a new output directory.")
        if finetuning_mode == "lora":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, checkpoint["adapter_state"])
        else: model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise ValueError(f"{resume_path} lacks optimizer state and cannot be safely resumed. Start a fresh run or use a newer checkpoint.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_path = output / "history.json"
        if history_path.exists(): history = json.loads(history_path.read_text())
        print(json.dumps({"resumed_from": str(resume_path), "completed_epoch": start_epoch - 1, "target_epochs": requested_epochs}))
    if start_epoch > requested_epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; requested total is {requested_epochs}. Nothing to train.")
        tensorboard.close()
        return
    for epoch in range(start_epoch, requested_epochs + 1):
        backbone_frozen = finetuning_mode == "full" and epoch <= freeze_backbone_epochs
        trainable_parameters = (sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) if finetuning_mode == "lora" else configure_trainable_layers(model, architecture, backbone_frozen))
        if epoch == start_epoch or epoch == freeze_backbone_epochs + 1:
            mode = "LoRA adapter fine-tuning" if finetuning_mode == "lora" else ("head-only warm-up" if backbone_frozen else "full fine-tuning")
            print(json.dumps({"epoch": epoch, "training_mode": mode, "trainable_parameters": trainable_parameters}))
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = _run_epoch(model, validation_loader, criterion, None, device, collect_per_class=True)
        mode_name = "lora" if finetuning_mode == "lora" else ("head_only_warmup" if backbone_frozen else "full_finetuning")
        record = {"epoch": epoch, "training_mode": mode_name, "trainable_parameters": trainable_parameters, "train": train_metrics, "validation": validation_metrics}
        history.append(record); print(json.dumps(record))
        if tensorboard.writer is not None:
            tensorboard.writer.add_scalar("classifier/train_loss", train_metrics["loss"], epoch)
            tensorboard.writer.add_scalar("classifier/train_accuracy", train_metrics["accuracy"], epoch)
            tensorboard.writer.add_scalar("classifier/validation_loss", validation_metrics["loss"], epoch)
            tensorboard.writer.add_scalar("classifier/validation_accuracy", validation_metrics["accuracy"], epoch)
            tensorboard.writer.add_scalar("classifier/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            tensorboard.writer.add_scalar("classifier/trainable_parameters", trainable_parameters, epoch)
            tensorboard.writer.add_scalar("classifier/backbone_frozen", int(backbone_frozen), epoch)
            for name, accuracy in validation_metrics["per_class_accuracy"].items():
                if accuracy is not None: tensorboard.writer.add_scalar(f"classifier_per_class_validation_accuracy/{name}", accuracy, epoch)
            tensorboard.writer.flush()
        payload = {"optimizer": optimizer.state_dict(), "epoch": epoch, "architecture": architecture, "finetuning_mode": finetuning_mode, "lora_rank": lora_rank, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "num_classes": len(CLASS_NAMES), "class_names": CLASS_NAMES, "config": cfg}
        if finetuning_mode == "lora":
            from peft import get_peft_model_state_dict
            payload["adapter_state"] = get_peft_model_state_dict(model)
        else: payload["model"] = model.state_dict()
        # Atomic replacement avoids corrupting last.pt if the process is interrupted while writing.
        temporary = output / "last.pt.tmp"
        torch.save(payload, temporary); os.replace(temporary, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    tensorboard.close()
    print(f"Saved classifier checkpoint: {output / 'last.pt'}")


if __name__ == "__main__": main()

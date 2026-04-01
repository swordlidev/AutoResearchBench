from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare import DATASET_DIR, TIME_BUDGET


# ---------------------------------------------------------------------------
# Pure PyTorch ViT Implementation
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Split image into patches and project to hidden_size dimension"""
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, hidden_size)
        return self.proj(x).flatten(2).transpose(1, 2)


class DropPath(nn.Module):
    """Stochastic Depth regularization"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x / keep_prob * mask


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = Attention(hidden_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Pure PyTorch Vision Transformer implementation"""
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 200,
        hidden_size: int = 192,
        intermediate_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 3,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.pos_drop = nn.Dropout(hidden_dropout_prob)

        # Linearly increasing drop path rate per layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_hidden_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                qkv_bias=qkv_bias,
                drop=hidden_dropout_prob,
                attn_drop=attention_probs_dropout_prob,
                drop_path=dpr[i],
                layer_norm_eps=layer_norm_eps,
            )
            for i in range(num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.head = nn.Linear(hidden_size, num_classes)

        # Weight initialization
        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range: float):
        nn.init.trunc_normal_(self.pos_embed, std=initializer_range)
        nn.init.trunc_normal_(self.cls_token, std=initializer_range)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, hidden_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, hidden_size)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # Use CLS token for classification

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    data_dir: str = DATASET_DIR
    image_size: int = 64
    patch_size: int = 8
    num_classes: int = 200
    hidden_size: int = 192
    intermediate_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 3
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    drop_path_rate: float = 0.0
    qkv_bias: bool = True
    batch_size: int = 256
    eval_batch_size: int = 256
    lr: float = 1e-3
    warmup_ratio: float = 0.05
    warmdown_ratio: float = 0.2
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    accumulation_steps: int = 1
    seed: int = 1337
    num_workers: int = 4
    device: str = "cuda"
    amp_dtype: str = "float16"
    eval_only: bool = False
    compile: bool = False
    log_interval: int = 150


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = buffer.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
    return tensor


def normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


class TrainTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        image = ImageOps.expand(image, border=4, fill=0)
        max_offset = image.size[0] - self.image_size
        left = random.randint(0, max_offset)
        top = random.randint(0, max_offset)
        image = image.crop((left, top, left + self.image_size, top + self.image_size))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return normalize(pil_to_tensor(image))


class EvalTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        return normalize(pil_to_tensor(image))


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.root = root
        self.transform = transform
        self.classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = root / class_name
            for image_path in sorted(class_dir.rglob("*.JPEG")):
                self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"no images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), target


def mixup_batch(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], lam


def build_model(config: TrainConfig) -> nn.Module:
    model = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=3,
        num_classes=config.num_classes,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
        initializer_range=config.initializer_range,
        drop_path_rate=config.drop_path_rate,
        qkv_bias=config.qkv_bias,
    )
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def get_lr_multiplier(
    progress: float, warmup_ratio: float, warmdown_ratio: float, final_lr_frac: float = 0.0
) -> float:
    """Return learning rate multiplier based on training progress (0~1), following nanochat's schedule."""
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio if warmdown_ratio > 0 else 0.0
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        preds = outputs.argmax(dim=-1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_examples += batch_size

    return {
        "val_loss": total_loss / total_examples,
        "val_acc1": total_correct / total_examples,
    }


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )


def get_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported amp dtype: {name}")
    return mapping[name]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT-Tiny on Tiny ImageNet")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--patch-size", type=int, default=TrainConfig.patch_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--warmdown-ratio", type=float, default=TrainConfig.warmdown_ratio)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--adam-beta1", type=float, default=TrainConfig.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=TrainConfig.adam_beta2)
    parser.add_argument("--adam-eps", type=float, default=TrainConfig.adam_eps)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    parser.add_argument("--accumulation-steps", type=int, default=TrainConfig.accumulation_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--amp-dtype", type=str, default=TrainConfig.amp_dtype)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--intermediate-size", type=int, default=TrainConfig.intermediate_size)
    parser.add_argument("--num-hidden-layers", type=int, default=TrainConfig.num_hidden_layers)
    parser.add_argument("--num-attention-heads", type=int, default=TrainConfig.num_attention_heads)
    parser.add_argument("--hidden-dropout-prob", type=float, default=TrainConfig.hidden_dropout_prob)
    parser.add_argument(
        "--attention-probs-dropout-prob",
        type=float,
        default=TrainConfig.attention_probs_dropout_prob,
    )
    parser.add_argument("--layer-norm-eps", type=float, default=TrainConfig.layer_norm_eps)
    parser.add_argument("--initializer-range", type=float, default=TrainConfig.initializer_range)
    parser.add_argument("--drop-path-rate", type=float, default=TrainConfig.drop_path_rate)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--qkv-bias", dest="qkv_bias", action="store_true")
    parser.add_argument("--no-qkv-bias", dest="qkv_bias", action="store_false")
    parser.add_argument("--compile", action="store_true")
    parser.set_defaults(qkv_bias=TrainConfig.qkv_bias)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = TrainConfig(**vars(args))
    if config.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    set_seed(config.seed)
    amp_dtype = get_amp_dtype(config.amp_dtype)

    data_root = Path(config.data_dir)
    train_root = data_root / "train"
    val_root = data_root / "val_split"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            f"dataset not prepared under {data_root}. Run `python prepare.py` first."
        )

    train_dataset = ImageFolderDataset(train_root, TrainTransform(config.image_size))
    val_dataset = ImageFolderDataset(val_root, EvalTransform(config.image_size))

    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError("train/val class folders do not match")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    print(f"train={len(train_dataset)} val={len(val_dataset)} cls={len(train_dataset.classes)} dev={device} steps/ep={len(train_loader)} budget={TIME_BUDGET}s")

    if config.eval_only:
        val_metrics = evaluate(model, val_loader, device)
        print(f"val_loss={val_metrics['val_loss']:.4f} val_acc1={val_metrics['val_acc1']:.4f}")
        return

    # ---------------------------------------------------------------------------
    # Time-budget-based training loop (following nanochat)
    # ---------------------------------------------------------------------------
    best_acc1 = 0.0
    global_step = 0
    epoch = 0
    total_training_time = 0.0
    warmup_steps = 10  # First few steps excluded from training time (compilation / data loading warmup)

    while total_training_time < TIME_BUDGET:
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(train_loader):
            t0 = time.time()

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images, targets_a, targets_b, lam = mixup_batch(images, targets, config.mixup_alpha)

            # Adjust learning rate based on time progress
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lr = config.lr * get_lr_multiplier(
                progress, config.warmup_ratio, config.warmdown_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                logits = outputs
                loss_a = F.cross_entropy(
                    logits, targets_a, label_smoothing=config.label_smoothing
                )
                loss_b = F.cross_entropy(
                    logits, targets_b, label_smoothing=config.label_smoothing
                )
                loss = lam * loss_a + (1.0 - lam) * loss_b
                loss_for_backward = loss / config.accumulation_steps

            scaler.scale(loss_for_backward).backward()

            should_step = (
                (step + 1) % config.accumulation_steps == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_examples += batch_size

            # Accumulate training time (skip warmup steps)
            torch.cuda.synchronize()
            dt = time.time() - t0
            if global_step > warmup_steps:
                total_training_time += dt

            if step % config.log_interval == 0:
                remaining = max(0, TIME_BUDGET - total_training_time)
                print(f"e{epoch+1:03d} s{step:04d}/{len(train_loader):04d} lr={lr:.6f} loss={loss.item():.4f} rem={remaining:.0f}s")

            # Early exit current epoch if time budget exceeded
            if global_step > warmup_steps and total_training_time >= TIME_BUDGET:
                break

        # Evaluate after epoch ends
        if epoch_examples > 0:
            train_metrics = {
                "train_loss": epoch_loss / epoch_examples,
                "train_acc1": epoch_correct / epoch_examples,
            }
            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - epoch_start

            record = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": elapsed,
                "total_training_time": total_training_time,
                **train_metrics,
                **val_metrics,
            }

            print(f"[ep{epoch+1:03d}] train_loss={record['train_loss']:.4f} train_acc1={record['train_acc1']:.4f} val_loss={record['val_loss']:.4f} val_acc1={record['val_acc1']:.4f} {elapsed:.1f}s {total_training_time:.0f}/{TIME_BUDGET}s")

            if record["val_acc1"] > best_acc1:
                best_acc1 = record["val_acc1"]

        epoch += 1

    print(f"best val acc1: {best_acc1:.4f}")
    print(f"total epochs: {epoch}")
    print(f"total steps: {global_step}")
    print(f"training time: {total_training_time:.1f}s")


if __name__ == "__main__":
    main()

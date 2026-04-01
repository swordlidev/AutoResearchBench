from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare import DATASET_DIR, TIME_BUDGET, tokenize


@dataclass
class TrainConfig:
    data_dir: str = DATASET_DIR
    time_budget_seconds: int = TIME_BUDGET
    vocab_size: int = 30000
    max_seq_length: int = 64
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    embedding_dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    num_classes: int = 2
    pooling: str = "mean"
    batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 3e-4
    warmup_ratio: float = 0.05
    warmdown_ratio: float = 0.2
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    token_dropout_prob: float = 0.0
    accumulation_steps: int = 1
    seed: int = 1337
    num_workers: int = 0
    device: str = "cuda"
    amp_dtype: str = "float16"
    lowercase: bool = True
    eval_only: bool = False
    compile: bool = False
    log_interval: int = 100
    eval_interval_seconds: float = 120.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported amp dtype: {name}")
    return mapping[name]


def get_lr_multiplier(
    progress: float, warmup_ratio: float, warmdown_ratio: float, final_lr_frac: float = 0.1
) -> float:
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    if progress < 1.0 - warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / warmdown_ratio if warmdown_ratio > 0 else 0.0
    return cooldown + (1.0 - cooldown) * final_lr_frac


class SST2Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        vocab: dict[str, int],
        max_seq_length: int,
        lowercase: bool,
        token_dropout_prob: float = 0.0,
        is_train: bool = False,
    ) -> None:
        self.samples: list[tuple[list[int], int]] = []
        self.token_dropout_prob = token_dropout_prob
        self.is_train = is_train
        cls_id = vocab["[CLS]"]
        unk_id = vocab["[UNK]"]

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["label"] is None:
                    continue
                tokens = tokenize(record["sentence"], lowercase=lowercase)
                token_ids = [cls_id]
                token_ids.extend(vocab.get(token, unk_id) for token in tokens)
                token_ids = token_ids[:max_seq_length]
                self.samples.append((token_ids, int(record["label"])))

        if not self.samples:
            raise RuntimeError(f"no labeled samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        token_ids, label = self.samples[index]
        if self.is_train and self.token_dropout_prob > 0.0 and len(token_ids) > 2:
            kept = [token_ids[0]]
            for token_id in token_ids[1:]:
                if random.random() >= self.token_dropout_prob:
                    kept.append(token_id)
            if len(kept) == 1:
                kept.append(token_ids[1])
            token_ids = kept
        return token_ids, label


def collate_batch(
    batch: list[tuple[list[int], int]],
    pad_id: int,
    max_seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch)
    lengths = [min(len(tokens), max_seq_length) for tokens, _ in batch]
    seq_len = max(lengths)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)

    for index, (tokens, label) in enumerate(batch):
        current = tokens[:seq_len]
        input_ids[index, : len(current)] = torch.tensor(current, dtype=torch.long)
        attention_mask[index, : len(current)] = True
        labels[index] = label
    return input_ids, attention_mask, labels


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.mlp(x))
        return x


class SmallTransformerClassifier(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.pooling = config.pooling
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        padding_mask = ~attention_mask
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        x = self.final_norm(x)
        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "mean":
            masked_x = x * attention_mask.unsqueeze(-1)
            pooled = masked_x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            raise ValueError(f"unsupported pooling: {self.pooling}")
        return self.classifier(pooled)


def build_model(config: TrainConfig) -> nn.Module:
    model = SmallTransformerClassifier(config)
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

        true_positive += ((preds == 1) & (labels == 1)).sum().item()
        false_positive += ((preds == 1) & (labels == 0)).sum().item()
        false_negative += ((preds == 0) & (labels == 1)).sum().item()

        precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "val_loss": total_loss / total_examples,
        "val_acc1": total_correct / total_examples,
        "val_recall": recall,
        "val_f1": f1,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SST-2 + Small Transformer")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--time-budget-seconds", type=int, default=TrainConfig.time_budget_seconds)
    parser.add_argument("--max-seq-length", type=int, default=TrainConfig.max_seq_length)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--intermediate-size", type=int, default=TrainConfig.intermediate_size)
    parser.add_argument("--num-hidden-layers", type=int, default=TrainConfig.num_hidden_layers)
    parser.add_argument("--num-attention-heads", type=int, default=TrainConfig.num_attention_heads)
    parser.add_argument("--embedding-dropout", type=float, default=TrainConfig.embedding_dropout)
    parser.add_argument("--hidden-dropout-prob", type=float, default=TrainConfig.hidden_dropout_prob)
    parser.add_argument(
        "--attention-probs-dropout-prob",
        type=float,
        default=TrainConfig.attention_probs_dropout_prob,
    )
    parser.add_argument("--classifier-dropout", type=float, default=TrainConfig.classifier_dropout)
    parser.add_argument("--layer-norm-eps", type=float, default=TrainConfig.layer_norm_eps)
    parser.add_argument("--pooling", type=str, default=TrainConfig.pooling)
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
    parser.add_argument("--token-dropout-prob", type=float, default=TrainConfig.token_dropout_prob)
    parser.add_argument("--accumulation-steps", type=int, default=TrainConfig.accumulation_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--amp-dtype", type=str, default=TrainConfig.amp_dtype)
    parser.add_argument("--eval-interval-seconds", type=float, default=TrainConfig.eval_interval_seconds)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.set_defaults(lowercase=TrainConfig.lowercase)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = TrainConfig(**vars(args))
    if config.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config.pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be one of: mean, cls")

    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp_dtype = get_amp_dtype(config.amp_dtype)
    set_seed(config.seed)

    data_root = Path(config.data_dir)
    train_path = data_root / "train.jsonl"
    val_path = data_root / "val.jsonl"
    vocab_path = data_root / "vocab.json"
    if not train_path.exists() or not val_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            f"dataset not prepared under {data_root}. Run `python prepare.py` first."
        )

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    config.vocab_size = len(vocab)
    pad_id = vocab["[PAD]"]

    train_dataset = SST2Dataset(
        train_path,
        vocab,
        config.max_seq_length,
        config.lowercase,
        token_dropout_prob=config.token_dropout_prob,
        is_train=True,
    )
    val_dataset = SST2Dataset(
        val_path,
        vocab,
        config.max_seq_length,
        config.lowercase,
        token_dropout_prob=0.0,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        collate_fn=lambda batch: collate_batch(batch, pad_id, config.max_seq_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        collate_fn=lambda batch: collate_batch(batch, pad_id, config.max_seq_length),
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    print(
        f"train={len(train_dataset)} val={len(val_dataset)} vocab={config.vocab_size} "
        f"dev={device} steps/ep={len(train_loader)} "
        f"seq={config.max_seq_length} d_model={config.hidden_size} "
        f"layers={config.num_hidden_layers} heads={config.num_attention_heads} "
        f"pool={config.pooling} tok_drop={config.token_dropout_prob:.2f} "
        f"budget={config.time_budget_seconds}s"
    )

    if config.eval_only:
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_acc1={val_metrics['val_acc1']:.4f} "
            f"val_recall={val_metrics['val_recall']:.4f} "
            f"val_f1={val_metrics['val_f1']:.4f}"
        )
        return

    best_acc1 = 0.0
    best_f1 = 0.0
    global_step = 0
    epoch = 0
    training_start = time.time()
    last_eval_time = training_start

    while time.time() - training_start < config.time_budget_seconds:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
            elapsed = time.time() - training_start
            if elapsed >= config.time_budget_seconds:
                break

            progress = min(elapsed / config.time_budget_seconds, 1.0)
            lr_mult = get_lr_multiplier(progress, config.warmup_ratio, config.warmdown_ratio)
            for group in optimizer.param_groups:
                group["lr"] = config.lr * lr_mult

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=config.label_smoothing,
                )
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

            batch_size = labels.size(0)
            epoch_examples += batch_size
            epoch_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()

            if should_step and global_step % config.log_interval == 0:
                print(
                    f"epoch {epoch:03d} "
                    f"step {step:04d}/{len(train_loader):04d} "
                    f"update {global_step:05d} "
                    f"lr {optimizer.param_groups[0]['lr']:.6f} "
                    f"loss {loss.item():.4f}"
                )

            now = time.time()
            if now - last_eval_time >= config.eval_interval_seconds:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"[eval] step={global_step:05d} "
                    f"min={(now - training_start) / 60.0:.2f} "
                    f"train_loss={epoch_loss / max(epoch_examples, 1):.4f} "
                    f"val_loss={val_metrics['val_loss']:.4f} "
                    f"val_acc1={val_metrics['val_acc1']:.4f} "
                    f"val_recall={val_metrics['val_recall']:.4f} "
                    f"val_f1={val_metrics['val_f1']:.4f}"
                )
                best_acc1 = max(best_acc1, val_metrics["val_acc1"])
                best_f1 = max(best_f1, val_metrics["val_f1"])
                last_eval_time = now
                model.train()

        if epoch_examples > 0:
            train_metrics = {
                "train_loss": epoch_loss / epoch_examples,
                "train_acc1": epoch_correct / epoch_examples,
            }
            val_metrics = evaluate(model, val_loader, device)
            best_acc1 = max(best_acc1, val_metrics["val_acc1"])
            best_f1 = max(best_f1, val_metrics["val_f1"])
            epoch_time = time.time() - epoch_start

            print(
                f"[ep{epoch:03d}] "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc1={train_metrics['train_acc1']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_acc1={val_metrics['val_acc1']:.4f} "
                f"val_recall={val_metrics['val_recall']:.4f} "
                f"val_f1={val_metrics['val_f1']:.4f} "
                f"{epoch_time:.1f}s {(time.time() - training_start):.0f}/{config.time_budget_seconds}s"
            )

    total_time = time.time() - training_start
    final_metrics = evaluate(model, val_loader, device)
    best_acc1 = max(best_acc1, final_metrics["val_acc1"])
    best_f1 = max(best_f1, final_metrics["val_f1"])
    print(f"best val acc1: {best_acc1:.4f}")
    print(f"best val f1: {best_f1:.4f}")
    print(f"final val loss: {final_metrics['val_loss']:.4f}")
    print(f"final val acc1: {final_metrics['val_acc1']:.4f}")
    print(f"final val recall: {final_metrics['val_recall']:.4f}")
    print(f"final val f1: {final_metrics['val_f1']:.4f}")
    print(f"total epochs: {epoch}")
    print(f"total steps: {global_step}")
    print(f"training time: {total_time:.1f}s")


if __name__ == "__main__":
    main()

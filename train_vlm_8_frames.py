"""
train_vlm_depth.py — Depth-aligned visual injection for InternVL3.5-1B.

Architecture (adapted from Depth_aligned_injection.py / MiniVLM v3):
  InternViT-300M (frozen, 24 blocks, hidden=1024):
    Intermediate features captured at blocks [7, 15, 23] (33%, 66%, 100%)
    via forward hooks during the single unified model() forward pass.
    Features are [B*N_frames, 1024_patches, 1024_hidden].
    Pooled from 1024 → 256 spatial tokens (32×32 → 16×16 adaptive avg pool)
    to match the CTX_PER_IMAGE=256 visual tokens used by InternVL's mlp1.

  Depth injection into Qwen3-0.6B (28 blocks):
    LLM block 0  post-hook: pool + project via DepthMLP, store injections
    LLM blocks [1, 2, 3] post-hooks: ADD projected features to visual prefix

  DepthMLP: Linear(1024→1024) + GELU + Linear(1024→1024)
    Near-zero output init (std=1e-3) so depth features start as tiny
    perturbations — training begins from the last-layer baseline.

  MultiTaskHead: pool at last non-padding token → step + stage classification

  Hook flow inside a single model() call:
    ViT block 7  → hook stores feats[7]   (33% depth)
    ViT block 15 → hook stores feats[15]  (66% depth)
    ViT block 23 → hook stores feats[23]  (100% depth, same as ViT last layer)
    LLM block 0  → post-hook: prepare_injections() reads feats, projects
    LLM block 1  → post-hook: ADD injections[0] to visual positions [bos:bos+2048]
    LLM block 2  → post-hook: ADD injections[1]
    LLM block 3  → post-hook: ADD injections[2]

Trainable:
  LoRA on Qwen3-0.6B (q/k/v/o_proj, r=8, alpha=16) — lr = 2e-4
  MultiTaskHead                                       — lr = 2e-4
  3× DepthMLP                                         — lr = 1e-4 (near-zero init)

Frozen:
  InternViT-300M (hooks only read, never modify)
  InternVL mlp1 projector

Run alongside train_vlm.py on GPU 0:
  python train_vlm_depth.py --config config.yaml --data_dir vlm_data/ --gpu 1
"""

import copy
import json
import os
import random
import time
import types
from pathlib import Path

# cuDNN broken by Mamba env-var changes — disable globally
import torch as _torch
_torch.backends.cudnn.enabled = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


# ── Constants ─────────────────────────────────────────────────────────────────

N_FRAMES      = 8
CTX_PER_IMAGE = 256        # visual tokens per frame after InternVL mlp1 (16×16)
IMG_SIZE      = 448
LLM_HIDDEN    = 1024       # Qwen3-0.6B hidden size
VIT_HIDDEN    = 1024       # InternViT-300M hidden size

# InternViT-300M: 24 transformer blocks (0-indexed 0…23).
# Depth layers at ~33%, 66%, 100%:
VIT_DEPTH_LAYERS  = [7, 15, 23]   # blocks 8, 16, 24 (1-indexed)
LLM_INJECT_BLOCKS = [1, 2, 3]     # Qwen3 LLM blocks to inject into

STEP_LABELS  = [f"step_{i:03d}" for i in range(0, 14)]   # 14 classes
STAGE_LABELS = ["stage_01", "stage_02", "stage_03"]       #  3 classes

LABEL2ID = {
    "step_classification":  {lb: i for i, lb in enumerate(STEP_LABELS)},
    "stage_classification": {lb: i for i, lb in enumerate(STAGE_LABELS)},
}

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

# DINOv2 ViT-L/14 (second encoder, separate forward pass)
DINO_HIDDEN       = 1024
DINO_IMG_SIZE     = 518   # native DINOv2 resolution → 37×37=1369 patches
DINO_DEPTH_LAYERS = [6, 12, 18]   # 25%, 50%, 75% of 24 blocks
DINO_LLM_BLOCKS   = [4,  8, 12]   # inject into Qwen3-0.6B blocks (28 total)
DINO_CTX_PER_FRAME = CTX_PER_IMAGE  # pool 37×37→16×16 = 256 tokens/frame (same as InternViT)

ALLOWED_TAGS = {"step_classification", "stage_classification"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def summarize(y_true, y_pred, exclude_class=None):
    if not y_true:
        return {"acc": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "kappa": 0.0, "n": 0}
    labels = None
    if exclude_class is not None:
        labels = [c for c in sorted(set(y_true) | set(y_pred)) if c != exclude_class]
    return {
        "acc":         accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro",
                                labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                labels=labels, zero_division=0),
        "kappa":       cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "n":           len(y_true),
    }


# ── Dataset statistics ────────────────────────────────────────────────────────

def dataset_stats(data_dir: Path) -> dict:
    """
    Print and return dataset statistics suitable for a research paper.
    Reads train/val/test JSON manifests from data_dir.
    """
    from collections import Counter as _C

    all_stats = {}
    print("\n" + "=" * 72)
    print("  DATASET STATISTICS")
    print("=" * 72)

    step_all, stage_all = _C(), _C()

    for split in ["train", "val", "test"]:
        p = data_dir / f"{split}.json"
        if not p.exists():
            print(f"  [{split}] not found: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            recs = json.load(f)

        subjects   = _C(r["meta"]["subject"] for r in recs)
        step_recs  = [r for r in recs if r["main_tag"] == "step_classification"]
        stage_recs = [r for r in recs if r["main_tag"] == "stage_classification"]
        step_dist  = _C(r["meta"]["step_id"]  for r in step_recs)
        stage_dist = _C(r["meta"]["stage_id"] for r in stage_recs)

        step_all.update(step_dist)
        stage_all.update(stage_dist)

        all_stats[split] = {
            "total_records":   len(recs),
            "step_records":    len(step_recs),
            "stage_records":   len(stage_recs),
            "n_subjects":      len(subjects),
            "step_dist":       dict(sorted(step_dist.items())),
            "stage_dist":      dict(sorted(stage_dist.items())),
        }

        print(f"\n  [{split.upper()}]")
        print(f"    Total records      : {len(recs):>6,}")
        print(f"    Step records       : {len(step_recs):>6,}")
        print(f"    Stage records      : {len(stage_recs):>6,}")
        print(f"    Unique subjects    : {len(subjects):>6,}")
        print(f"    Step distribution  :")
        for k, v in sorted(step_dist.items()):
            bar = "█" * int(v / max(step_dist.values()) * 30)
            print(f"      {k}: {v:>5,}  {bar}")
        print(f"    Stage distribution :")
        for k, v in sorted(stage_dist.items()):
            bar = "█" * int(v / max(stage_dist.values()) * 30)
            print(f"      {k}: {v:>5,}  {bar}")

    # overall
    if step_all:
        imb = max(step_all.values()) / max(min(step_all.values()), 1)
        print(f"\n  [ALL SPLITS — STEPS]")
        print(f"    Classes            : {len(step_all)}")
        print(f"    Imbalance ratio    : {imb:.1f}×")
        print(f"    Most common        : {step_all.most_common(1)[0]}")
        print(f"    Least common       : {step_all.most_common()[-1]}")
    print("=" * 72 + "\n")
    return all_stats


# ── Full per-class report + confusion matrix ──────────────────────────────────

def full_report(y_true: list, y_pred: list,
                int_to_label: dict, task_name: str,
                out_dir: Path | None = None,
                exclude_class: int | None = None) -> dict:
    """
    Print classification_report + confusion matrix.
    Saves CSV and JSON files if out_dir is provided.
    Returns dict with all metrics for JSON serialisation.
    """
    if not y_true:
        print(f"[{task_name}] No predictions collected.")
        return {}

    labels_present = sorted(set(y_true) | set(y_pred))
    if exclude_class is not None:
        labels_present = [l for l in labels_present if l != exclude_class]

    names = [int_to_label.get(i, str(i)) for i in labels_present]

    print(f"\n{'─'*72}")
    print(f"  {task_name.upper()}  —  per-class report")
    print(f"{'─'*72}")
    report_str = classification_report(
        y_true, y_pred,
        labels=labels_present, target_names=names,
        zero_division=0,
    )
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    header = "           " + "  ".join(f"{n:>8}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {names[i]:>9}  {row_str}")

    report_dict = classification_report(
        y_true, y_pred,
        labels=labels_present, target_names=names,
        output_dict=True, zero_division=0,
    )

    result = {
        "classification_report": report_dict,
        "confusion_matrix":      cm.tolist(),
        "label_order":           names,
        "accuracy":              accuracy_score(y_true, y_pred),
        "macro_f1":              f1_score(y_true, y_pred, average="macro",
                                          labels=labels_present, zero_division=0),
        "weighted_f1":           f1_score(y_true, y_pred, average="weighted",
                                          labels=labels_present, zero_division=0),
        "kappa":                 cohen_kappa_score(y_true, y_pred)
                                 if len(set(y_true)) > 1 else 0.0,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = task_name.replace(" ", "_").lower()
        # Save confusion matrix as CSV
        import csv
        cm_path = out_dir / f"confusion_matrix_{tag}.csv"
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for i, row in enumerate(cm):
                w.writerow([names[i]] + list(row))
        # Save full report as JSON
        rpt_path = out_dir / f"report_{tag}.json"
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {cm_path.name}  {rpt_path.name}")

    return result


# ── DepthMLP ──────────────────────────────────────────────────────────────────

class DepthMLP(nn.Module):
    """
    Projects InternViT intermediate features into LLM hidden space.

    Near-zero output init (std=1e-3) ensures depth features start as tiny
    perturbations — training begins from the last-layer-only baseline and
    smoothly introduces depth information via gradient updates.

    Input:  [B, N_tokens, vit_hidden]
    Output: [B, N_tokens, llm_hidden]
    """

    def __init__(self, vit_hidden: int = VIT_HIDDEN, llm_hidden: int = LLM_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        # Near-zero init on output layer — depth starts as ~zero perturbation
        nn.init.normal_(self.net[2].weight, std=1e-3)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.net[0].weight.dtype))


# ── DINOv2 loader ─────────────────────────────────────────────────────────────

def load_dino(dino_path: str, device: torch.device):
    """Load frozen DINOv2 ViT-L/14 with native AutoImageProcessor."""
    print(f"[DINOv2] Loading processor: {dino_path}")
    proc = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    print(f"[DINOv2] Loading model ...")
    dino = AutoModel.from_pretrained(
        dino_path, local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # cuDNN globally disabled; eager is safest
    ).to(device).eval()
    dino.requires_grad_(False)
    n = sum(p.numel() for p in dino.parameters())
    print(f"[DINOv2] Ready: hidden={dino.config.hidden_size}  "
          f"layers={dino.config.num_hidden_layers}  params={n/1e6:.1f}M  "
          f"device={device}")
    return dino, proc


# ── DINOv2Injector (second encoder, separate forward pass) ────────────────────

class DINOv2Injector(nn.Module):
    """
    Second vision encoder: DINOv2 ViT-L/14 features injected into Qwen3-0.6B.

    Runs a SEPARATE DINOv2 forward pass BEFORE the InternVL model() call:
      1. prepare_injections(): load 8 frames at 518×518, run DINOv2,
         capture hooks at blocks [6,12,18], pool 37×37→16×16 (256 tok/frame),
         project via DepthMLP, store in _injections.
      2. Permanent LLM hooks on blocks [4,8,12]: ADD stored injections.
         (Complements InternViT injection at blocks [1,2,3].)

    The two encoders inject at non-overlapping LLM blocks:
      InternViT → LLM blocks [1,2,3]  (early, high-res RGB features)
      DINOv2    → LLM blocks [4,8,12] (mid-range, semantic depth features)
    """

    def __init__(self, dino_model: nn.Module, dino_processor,
                 vit_layers: list = DINO_DEPTH_LAYERS,
                 llm_blocks: list = DINO_LLM_BLOCKS,
                 dino_hidden: int = DINO_HIDDEN,
                 llm_hidden:  int = LLM_HIDDEN,
                 n_frames: int = N_FRAMES,
                 img_size: int = DINO_IMG_SIZE,
                 ctx_per_frame: int = DINO_CTX_PER_FRAME):
        super().__init__()
        assert len(vit_layers) == len(llm_blocks)
        self._dino_processor = dino_processor
        self.vit_layers    = sorted(vit_layers)
        self.llm_blocks    = llm_blocks
        self.n_frames      = n_frames
        self.img_size      = img_size
        self.ctx_per_frame = ctx_per_frame
        self.n_visual      = n_frames * ctx_per_frame

        self.depth_mlps = nn.ModuleList([
            DepthMLP(dino_hidden, llm_hidden) for _ in range(len(vit_layers))
        ])

        self._dino_blocks = list(dino_model.encoder.layer)
        self._vit_feats:  dict = {}
        self._vit_hooks:  list = []
        self._llm_hooks:  list = []
        self._injections: dict = {}

        for p in dino_model.parameters():
            p.requires_grad = False
        dino_model.eval()

        print(f"[DINOv2Injector] ViT layers {self.vit_layers} → "
              f"LLM blocks {self.llm_blocks}")
        print(f"[DINOv2Injector] {dino_hidden}→{llm_hidden}  "
              f"{n_frames}×{img_size}px  pool→{ctx_per_frame} tok/frame")

    # ── ViT hooks (attached/detached per DINOv2 forward) ──────────────────────

    def _attach_vit_hooks(self, dino_model: nn.Module):
        self._detach_vit_hooks()
        self._vit_feats.clear()
        for layer_idx in self.vit_layers:
            def _hook(module, inp, out, _idx=layer_idx):
                h = out[0] if isinstance(out, tuple) else out
                self._vit_feats[_idx] = h[:, 1:, :].detach()  # drop CLS
            self._vit_hooks.append(
                self._dino_blocks[layer_idx].register_forward_hook(_hook))

    def _detach_vit_hooks(self):
        for h in self._vit_hooks: h.remove()
        self._vit_hooks.clear()

    # ── LLM hooks (registered once, permanent) ────────────────────────────────

    def register_llm_hooks(self, language_model: nn.Module):
        """Register ADD hooks on Qwen3-0.6B at DINO_LLM_BLOCKS. Call once."""
        for h in self._llm_hooks: h.remove()
        self._llm_hooks.clear()

        # Navigate through PeftModel wrapper → Qwen3ForCausalLM → .model → .layers
        raw = language_model
        visited = set()
        while hasattr(raw, "base_model"):
            if id(raw) in visited: break
            visited.add(id(raw)); raw = raw.base_model
        if hasattr(raw, "model") and not hasattr(raw, "layers"):
            raw = raw.model
        if not hasattr(raw, "layers"):
            raise RuntimeError(
                f"[DINOv2Injector] Cannot find .layers on LLM. "
                f"Attrs: {[a for a in dir(raw) if 'layer' in a.lower()]}"
            )
        blocks = raw.layers
        print(f"[DINOv2Injector] Qwen3-0.6B: {len(blocks)} blocks — "
              f"registering hooks at {self.llm_blocks}")

        for depth_idx, block_idx in enumerate(self.llm_blocks):
            if block_idx >= len(blocks):
                raise ValueError(
                    f"DINOv2 LLM block {block_idx} >= total {len(blocks)}")

            def _hook(module, inp, out, _d=depth_idx):
                payload = self._injections.get(_d)
                if payload is None: return out
                hidden = out[0] if isinstance(out, tuple) else out
                B, T, H = hidden.shape
                inj     = payload.to(dtype=hidden.dtype, device=hidden.device)
                add_len = min(T, inj.shape[1])
                updated = hidden.clone()
                updated[:, :add_len, :] = hidden[:, :add_len, :] + inj[:, :add_len, :]
                return (updated,) + out[1:] if isinstance(out, tuple) else updated

            self._llm_hooks.append(
                blocks[block_idx].register_forward_hook(_hook))

        print(f"[DINOv2Injector] LLM hooks registered at blocks {self.llm_blocks}")

    # ── Pre-pass (call before InternVL model() call) ──────────────────────────

    @torch.no_grad()
    def prepare_injections(self, dino_model: nn.Module,
                            image_paths: list, B: int):
        """
        Run DINOv2 at 518×518 on B*N_frames images, pool and project.
        image_paths: flat list length B*N_frames
        """
        self._vit_feats.clear(); self._injections.clear()
        BN = B * self.n_frames
        self._attach_vit_hooks(dino_model)

        images = []
        for p in image_paths:
            try:   images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB",
                    (self.img_size, self.img_size), (128, 128, 128)))

        dev    = next(dino_model.parameters()).device
        inputs = self._dino_processor(
            images=images, return_tensors="pt",
            size={"height": self.img_size, "width": self.img_size},
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        dino_model(**inputs)
        self._detach_vit_hooks()

        sqrt_ctx = int(round(self.ctx_per_frame ** 0.5))  # 16
        for depth_idx, layer_idx in enumerate(self.vit_layers):
            feats = self._vit_feats.get(layer_idx)
            if feats is None: continue
            # feats: [B*N, 1369, 1024] (CLS dropped in hook)
            BN_f, n_patches, D = feats.shape
            sqrt_p  = int(round(n_patches ** 0.5))
            spatial = feats.permute(0, 2, 1).reshape(BN_f, D, sqrt_p, sqrt_p)
            pooled  = F.adaptive_avg_pool2d(
                spatial.float(), output_size=(sqrt_ctx, sqrt_ctx))
            flat    = pooled.reshape(BN_f, D, self.ctx_per_frame).permute(0, 2, 1)
            flat    = flat.to(feats.dtype)
            combined  = flat.reshape(B, self.n_visual, D)          # [B, N_VIS, 1024]
            projected = self.depth_mlps[depth_idx](combined)        # [B, N_VIS, llm_hidden]
            self._injections[depth_idx] = projected

    def clear_state(self):
        self._vit_feats.clear(); self._injections.clear()


# ── InternViTDepthInjector ────────────────────────────────────────────────────

class InternViTDepthInjector(nn.Module):
    """
    Depth-aligned visual injection for InternVL3.5-1B.

    Captures InternViT intermediate layer features via forward hooks during the
    single unified InternVL model() forward pass, then injects projected
    features into early Qwen3-0.6B LLM blocks via elementwise ADD.

    Design choices (adapted from Depth_aligned_injection.py):
      - Hooks on frozen ViT blocks — no second forward pass needed
      - Pool 1024 → 256 spatial tokens per frame via 2D adaptive avg pool
        (32×32 → 16×16) to match CTX_PER_IMAGE used by InternVL's mlp1
      - LLM block 0 post-hook prepares injections (runs after all ViT features
        are captured but before the LLM's own transformer reasoning begins)
      - ADD (not concat) — no extra sequence length or architecture change
      - Separate lower LR for DepthMLPs (near-zero init)
    """

    def __init__(
        self,
        vision_model: nn.Module,
        vit_depth_layers: list = VIT_DEPTH_LAYERS,
        llm_inject_blocks: list = LLM_INJECT_BLOCKS,
        vit_hidden: int = VIT_HIDDEN,
        llm_hidden: int = LLM_HIDDEN,
        n_frames: int = N_FRAMES,
    ):
        super().__init__()
        assert len(vit_depth_layers) == len(llm_inject_blocks), \
            "Need one LLM block per ViT depth layer"

        self.vit_depth_layers  = sorted(vit_depth_layers)
        self.llm_inject_blocks = llm_inject_blocks
        self.vit_hidden        = vit_hidden
        self.llm_hidden        = llm_hidden
        self.n_frames          = n_frames
        self.n_visual          = n_frames * CTX_PER_IMAGE  # 2048 visual positions in LLM

        self.depth_mlps = nn.ModuleList([
            DepthMLP(vit_hidden, llm_hidden) for _ in range(len(vit_depth_layers))
        ])

        # Per-batch state (cleared after each forward pass)
        self._vit_feats:  dict = {}   # {layer_idx: Tensor[B*N, n_patches, vit_hidden]}
        self._injections: dict = {}   # {depth_idx: Tensor[B, n_visual, llm_hidden]}

        # Hook handles (for cleanup)
        self._vit_hooks: list = []
        self._llm_hooks: list = []

        # Discover and cache encoder blocks once at construction — avoids the
        # costly named_modules() traversal and noisy print on every batch.
        self._encoder_blocks = self._find_encoder_blocks(vision_model)

    # ── ViT encoder block discovery (called once at __init__) ─────────────────

    @staticmethod
    def _find_encoder_blocks(vision_model: nn.Module):
        """Navigate InternViT to find the list of transformer blocks."""
        for path in [("encoder", "layers"), ("encoder", "layer")]:
            obj = vision_model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                print(f"[DepthInjector] ViT encoder at "
                      f".{'.' .join(path)}  n_blocks={len(obj)}")
                return obj
        # Fallback: first ModuleList with > 4 entries
        for name, mod in vision_model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 4:
                print(f"[DepthInjector] Fallback ViT encoder: "
                      f".{name}  n_blocks={len(mod)}")
                return mod
        raise RuntimeError(
            "[DepthInjector] Cannot find InternViT encoder blocks. "
            f"Top-level attrs: "
            f"{[a for a in dir(vision_model) if not a.startswith('_')]}"
        )

    # ── ViT hooks (attached/detached per batch) ────────────────────────────────

    def attach_vit_hooks(self):
        """Attach forward hooks to ViT depth layers. Call before model()."""
        self.detach_vit_hooks()
        self._vit_feats.clear()

        for layer_idx in self.vit_depth_layers:
            def _hook(module, inp, out, _idx=layer_idx):
                hidden = out[0] if isinstance(out, tuple) else out
                # Detach: gradients flow through DepthMLP output, not through ViT
                self._vit_feats[_idx] = hidden.detach()
            self._vit_hooks.append(
                self._encoder_blocks[layer_idx].register_forward_hook(_hook)
            )

    def detach_vit_hooks(self):
        for h in self._vit_hooks:
            h.remove()
        self._vit_hooks.clear()

    # ── LLM hooks (registered once after LoRA wrapping) ───────────────────────

    def register_llm_hooks(self, language_model: nn.Module, bos_offset: int = 0):
        """
        Register permanent hooks on Qwen3-0.6B transformer blocks.
        Call once after LoRA wrapping.

        Block 0 post-hook: reads _vit_feats, pools 1024→256 tokens/frame,
                           projects via DepthMLP, stores in _injections.
        Blocks [1,2,3] post-hooks: ADD _injections to visual prefix positions.
        """
        for h in self._llm_hooks:
            h.remove()
        self._llm_hooks.clear()

        # Unwrap PeftModel layers to reach raw Qwen3 model
        raw = language_model
        visited = set()
        while hasattr(raw, "base_model"):
            if id(raw) in visited:
                break
            visited.add(id(raw))
            raw = raw.base_model

        # Qwen3ForCausalLM wraps the decoder as .model
        if hasattr(raw, "model") and not hasattr(raw, "layers"):
            raw = raw.model

        if not hasattr(raw, "layers"):
            raise RuntimeError(
                "[DepthInjector] Cannot find .layers on LLM. "
                f"Available attrs with 'layer': "
                f"{[a for a in dir(raw) if 'layer' in a.lower()]}"
            )

        blocks = raw.layers
        n_blocks = len(blocks)
        print(f"[DepthInjector] Qwen3 LLM: {n_blocks} blocks  "
              f"bos_offset={bos_offset}")

        # ── Block 0 post-hook: prepare injections from cached ViT features ────
        def _prepare_hook(module, inp, out):
            self._prepare_injections()
            return out  # pass through unchanged

        self._llm_hooks.append(
            blocks[0].register_forward_hook(_prepare_hook)
        )

        # ── Blocks 1, 2, 3: ADD injections to visual prefix positions ─────────
        _start  = bos_offset
        _n_vis  = self.n_visual

        for depth_idx, block_idx in enumerate(self.llm_inject_blocks):
            if block_idx >= n_blocks:
                raise ValueError(
                    f"LLM inject block {block_idx} >= n_blocks {n_blocks}"
                )

            def _inject_hook(module, inp, out,
                             _d=depth_idx, _s=_start, _n=_n_vis):
                payload = self._injections.get(_d)
                if payload is None:
                    return out  # no injection if prepare_injections was skipped
                hidden = out[0] if isinstance(out, tuple) else out
                B, T, H = hidden.shape
                inj = payload.to(dtype=hidden.dtype, device=hidden.device)
                # ADD to visual prefix positions [bos_offset : bos_offset+n_visual]
                add_len = min(_n, T - _s)
                if add_len <= 0:
                    return out
                updated = hidden.clone()
                updated[:, _s:_s + add_len, :] = (
                    hidden[:, _s:_s + add_len, :] + inj[:, :add_len, :]
                )
                return (updated,) + out[1:] if isinstance(out, tuple) else updated

            self._llm_hooks.append(
                blocks[block_idx].register_forward_hook(_inject_hook)
            )

        print(f"[DepthInjector] LLM hooks registered: "
              f"prepare@block0  inject@blocks {self.llm_inject_blocks}")

    def remove_llm_hooks(self):
        for h in self._llm_hooks:
            h.remove()
        self._llm_hooks.clear()

    # ── Injection preparation (called from LLM block 0 post-hook) ─────────────

    def _prepare_injections(self):
        """
        Pool ViT intermediate features from n_patches → CTX_PER_IMAGE tokens
        per frame using 2D adaptive average pool (32×32 → 16×16), reshape to
        [B, n_visual, vit_hidden], and project via each DepthMLP.

        Must run AFTER ViT forward (feats cached in _vit_feats) and BEFORE
        LLM blocks [1,2,3] (which consume the prepared injections).
        Called automatically from the LLM block 0 post-hook.
        """
        self._injections.clear()
        if not self._vit_feats:
            return  # ViT hooks not attached (e.g. evaluation without injection)

        first = next(iter(self._vit_feats.values()))
        BN = first.shape[0]
        B  = BN // self.n_frames
        if B <= 0 or B * self.n_frames != BN:
            return  # unexpected shape — skip silently

        for depth_idx, layer_idx in enumerate(self.vit_depth_layers):
            feats = self._vit_feats.get(layer_idx)
            if feats is None:
                continue

            BN, n_tokens, D = feats.shape

            # Drop CLS token if present — InternViT includes CLS at position 0,
            # giving 1025 tokens (1 CLS + 32×32 patches). Detect by checking
            # whether n_tokens is a perfect square.
            sqrt_p = int(round(n_tokens ** 0.5))
            if sqrt_p * sqrt_p != n_tokens:
                feats  = feats[:, 1:, :]          # drop CLS → [B*N, 1024, D]
                n_tokens = feats.shape[1]
                sqrt_p   = int(round(n_tokens ** 0.5))

            # [B*N, n_patches, D] → 2D spatial → pool to 16×16 → [B*N, 256, D]
            spatial = feats.permute(0, 2, 1).reshape(BN, D, sqrt_p, sqrt_p)
            pooled  = F.adaptive_avg_pool2d(
                spatial.float(), output_size=(16, 16)
            )                                                 # [B*N, D, 16, 16]
            flat = pooled.reshape(BN, D, CTX_PER_IMAGE).permute(0, 2, 1)
            flat = flat.to(feats.dtype)                       # [B*N, 256, D]

            # Reshape frames together: [B, n_frames*256, D] = [B, n_visual, D]
            combined  = flat.reshape(B, self.n_visual, D)

            # Project via DepthMLP: [B, n_visual, llm_hidden]
            projected = self.depth_mlps[depth_idx](combined)
            self._injections[depth_idx] = projected

    def clear_state(self):
        """Clear per-batch state after each forward pass."""
        self._vit_feats.clear()
        self._injections.clear()


# ── Model loading (same as train_vlm.py) ──────────────────────────────────────

def load_model_and_tokenizer(model_path: str, gpu_id: int):
    device_str = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    import transformers.modeling_utils as _mu
    _orig_gtbc = getattr(_mu, "get_total_byte_count", None)
    if _orig_gtbc is not None:
        def _patched_gtbc(model, device_map, hf_quantizer=None):
            if not hasattr(model, "all_tied_weights_keys"):
                object.__setattr__(model, "all_tied_weights_keys", {})
            return _orig_gtbc(model, device_map, hf_quantizer)
        _mu.get_total_byte_count = _patched_gtbc

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device_str,
    )
    model.eval()

    _vp = next(model.vision_model.parameters())
    print(f"vision_model: device={_vp.device}  dtype={_vp.dtype}")

    # cuDNN patch: disable cuDNN for the patch_embedding conv2d (bf16 issue)
    _pe = model.vision_model.embeddings.patch_embedding

    def _fp32_conv_forward(self, x):
        orig = x.dtype
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            out = torch.nn.functional.conv2d(
                x.to(torch.float32),
                self.weight.to(torch.float32),
                self.bias.to(torch.float32) if self.bias is not None else None,
                self.stride, self.padding, self.dilation, self.groups,
            )
        finally:
            torch.backends.cudnn.enabled = prev
        return out.to(orig)

    import types as _types
    _pe.forward = _types.MethodType(_fp32_conv_forward, _pe)
    print("✅ patch_embedding patched: no-cuDNN fp32 conv")

    img_ctx_id = None
    for tok in ["<IMG_CONTEXT>", "<img_context>", "<image_context>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            img_ctx_id = tid
            break
    if img_ctx_id is None:
        img_ctx_id = 151671
        print(f"⚠  img_context_token_id not found, using fallback {img_ctx_id}")
    else:
        print(f"✅ img_context_token_id = {img_ctx_id}")
    model.img_context_token_id = img_ctx_id

    print(f"Loaded: {type(model).__name__} | LLM: {type(model.language_model).__name__}")
    print(f"  bos_token_id={tokenizer.bos_token_id}  eos_token_id={tokenizer.eos_token_id}")
    return model, tokenizer


def attach_lora(model):
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 8,
        lora_alpha     = 16,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias           = "none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()

    # Freeze vision encoder and mlp1 — only LoRA + DepthMLPs + head train
    model.vision_model.requires_grad_(False)
    if hasattr(model, "mlp1"):
        model.mlp1.requires_grad_(False)

    if not hasattr(model.language_model, "_orig_prepare_inputs_for_generation"):
        model.language_model._orig_prepare_inputs_for_generation = \
            model.language_model.prepare_inputs_for_generation

    def safe_prepare(self, input_ids, past_key_values=None, attention_mask=None,
                     inputs_embeds=None, **kwargs):
        bad = False
        try:
            if past_key_values is not None:
                if len(past_key_values) == 0:
                    bad = True
                elif past_key_values[0] is None:
                    bad = True
                elif isinstance(past_key_values[0], (tuple, list)) and past_key_values[0][0] is None:
                    bad = True
        except Exception:
            bad = True
        if bad:
            past_key_values = None
        return self._orig_prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs,
        )

    model.language_model.prepare_inputs_for_generation = types.MethodType(
        safe_prepare, model.language_model
    )
    return model


# ── Classification head ───────────────────────────────────────────────────────

class MultiTaskHead(nn.Module):
    def __init__(self, hidden: int = LLM_HIDDEN, n_step: int = 14, n_stage: int = 3):
        super().__init__()
        self.step  = nn.Linear(hidden, n_step)
        self.stage = nn.Linear(hidden, n_stage)

    def forward(self, x):
        return {"step_logits": self.step(x), "stage_logits": self.stage(x)}


# ── Dataset (same as train_vlm.py) ───────────────────────────────────────────

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def build_inputs(tokenizer, n_frames: int, tag: str, img_ctx_id: int):
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    n_ctx = n_frames * CTX_PER_IMAGE

    prompt     = "Classify step." if tag == "step_classification" else "Classify stage."
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    ids = []
    if bos is not None:
        ids.append(bos)
    ids += [img_ctx_id] * n_ctx + prompt_ids
    if eos is not None:
        ids.append(eos)

    input_ids      = torch.tensor(ids,         dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    image_flags    = torch.ones((n_frames,),    dtype=torch.long)
    return input_ids, attention_mask, image_flags


class SurgicalDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, img_ctx_id: int,
                 frames_base: str = ""):
        with open(json_path, encoding="utf-8") as f:
            all_samples = json.load(f)
        self.samples    = [s for s in all_samples if s.get("main_tag") in ALLOWED_TAGS]
        self.tokenizer  = tokenizer
        self.img_ctx_id = img_ctx_id
        # frames_base: prepended to relative frame paths recorded in the JSON.
        # Use when the JSON was built from a different working directory.
        self.frames_base = frames_base

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        tag = s["main_tag"]
        meta = s.get("meta", {}) or {}

        label_str = meta["step_id"] if tag == "step_classification" else meta["stage_id"]
        y         = LABEL2ID[tag][label_str]

        frames = s["frames"][:N_FRAMES]
        while len(frames) < N_FRAMES:
            frames.append(frames[-1])

        # Resolve relative paths against frames_base (empty = use as-is)
        if self.frames_base:
            frames = [
                p if os.path.isabs(p) else os.path.join(self.frames_base, p)
                for p in frames
            ]

        pixel_values = torch.stack(
            [img_transform(Image.open(p).convert("RGB")) for p in frames]
        )

        input_ids, attention_mask, image_flags = build_inputs(
            self.tokenizer, N_FRAMES, tag, self.img_ctx_id
        )

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image_flags":    image_flags,
            "frames":         frames,   # raw paths for DINOv2 518×518 pre-pass
            "tag":            tag,
            "y":              y,
        }


def collate_fn(batch, pad_id=0):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    def pad(x, v):
        if x.shape[0] == max_len:
            return x
        return torch.cat([x, torch.full((max_len - x.shape[0],), v, dtype=x.dtype)])

    return {
        "pixel_values":   torch.stack([b["pixel_values"]                      for b in batch]),
        "input_ids":      torch.stack([pad(b["input_ids"],      pad_id)        for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0)             for b in batch]),
        "image_flags":    torch.stack([b["image_flags"]                        for b in batch]),
        "frames":         [p for b in batch for p in b["frames"]],  # flat [B*N_FRAMES]
        "tags":           [b["tag"] for b in batch],
        "y":              torch.tensor([b["y"] for b in batch], dtype=torch.long),
    }


# ── Forward pass (depth-injected) ────────────────────────────────────────────

def forward_pooled_depth(batch, model, head, device,
                          vit_injector, dino_model=None, dino_injector=None):
    """
    Dual-encoder forward pass: InternViT depth + DINOv2 depth.

    Flow:
      1. DINOv2 pre-pass (separate): load 518×518 frames → hook layers [6,12,18]
         → pool 37×37→16×16 → project → store _injections (fires at LLM [4,8,12])
      2. Attach InternViT hooks (fire inside vision_model() → LLM [1,2,3])
      3. model() — both injection sets fire as LLM blocks execute
      4. Detach InternViT hooks; clear DINOv2 state
      5. Pool last non-padding token → classify
    """
    B, N, C, H, W = batch["pixel_values"].shape

    # ── Step 1: DINOv2 pre-pass (if second encoder is active) ────────────────
    if dino_model is not None and dino_injector is not None:
        dino_injector.prepare_injections(dino_model, batch["frames"], B)

    # ── Step 2: InternVL forward ──────────────────────────────────────────────
    pv             = batch["pixel_values"].to(device=device, dtype=torch.bfloat16).view(B * N, C, H, W)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    image_flags    = batch["image_flags"].to(device).view(B * N)

    vit_injector.attach_vit_hooks()

    out = model(
        pixel_values         = pv,
        input_ids            = input_ids,
        attention_mask       = attention_mask,
        image_flags          = image_flags,
        output_hidden_states = True,
        return_dict          = True,
        use_cache            = False,
    )

    vit_injector.detach_vit_hooks()

    # ── Step 3: Pool last token → classify ───────────────────────────────────
    seq_lens = attention_mask.sum(dim=1) - 1
    pooled   = out.hidden_states[-1][
        torch.arange(B, device=device), seq_lens, :
    ].float()
    logits = head(pooled)

    vit_injector.clear_state()
    if dino_injector is not None:
        dino_injector.clear_state()

    return logits, batch["tags"], batch["y"].to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, head, loader, device, vit_injector,
             dino_model=None, dino_injector=None, desc="eval"):
    model.eval(); head.eval(); vit_injector.eval()
    if dino_injector is not None: dino_injector.eval()
    ys_step, ps_step, ys_stage, ps_stage = [], [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        logits, tags, y = forward_pooled_depth(
            batch, model, head, device, vit_injector, dino_model, dino_injector)
        tag   = tags[0]
        ytrue = int(y.item())

        if tag == "step_classification":
            pred = int(torch.argmax(logits["step_logits"],  dim=-1).item())
            ys_step.append(ytrue); ps_step.append(pred)
        else:
            pred = int(torch.argmax(logits["stage_logits"], dim=-1).item())
            ys_stage.append(ytrue); ps_stage.append(pred)

    step_res  = summarize(ys_step,  ps_step,  exclude_class=0)
    stage_res = summarize(ys_stage, ps_stage)
    # also return raw lists for per-class analysis at test time
    return step_res, stage_res, ys_step, ps_step, ys_stage, ps_stage


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="/mnt/share/ali/surgical_assessment_v2/Retry/config.yaml")
    parser.add_argument("--data_dir",    default="/mnt/share/ali/surgical_assessment_v2/Retry")
    parser.add_argument("--frames_base", default="",
                        help="Prepend to relative frame paths in the JSON "
                             "(leave empty when build_vlm_data.py wrote absolute paths)")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--gpu",      type=int, default=1,
                        help="GPU id (default 1 — run alongside train_vlm.py on GPU 0)")
    parser.add_argument("--lora_lr",      type=float, default=2e-4,
                        help="Learning rate for LoRA + head")
    parser.add_argument("--dino_model",  required=True, help="Path to DINOv2 model directory")
    parser.add_argument("--depth_mlp_lr", type=float, default=1e-4,
                        help="Learning rate for DepthMLPs (near-zero init → lower LR)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"]  = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Suppress harmless "None of the inputs have requires_grad=True" warning
    # that fires from ViT gradient checkpointing when the ViT is frozen.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="None of the inputs have requires_grad=True",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*use_reentrant.*",
        category=UserWarning,
    )

    set_seed(42)
    cfg    = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}")

    model_path = cfg["data"]["internvl_model"]
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"]) / "depth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  InternVL3.5-1B — Dual Encoder: InternViT + DINOv2-L Depth Injection")
    print("=" * 72)
    print(f"  InternViT-300M  : 24 blocks, hidden={VIT_HIDDEN}")
    print(f"    Depth layers  : {[d+1 for d in VIT_DEPTH_LAYERS]} → LLM blocks {LLM_INJECT_BLOCKS}")
    print(f"  DINOv2 ViT-L/14 : 24 blocks, hidden={DINO_HIDDEN}, 518×518")
    print(f"    Depth layers  : {[d+1 for d in DINO_DEPTH_LAYERS]} → LLM blocks {DINO_LLM_BLOCKS}")
    print(f"  Frames          : {N_FRAMES} × {CTX_PER_IMAGE} = {N_FRAMES*CTX_PER_IMAGE} tokens/sample")
    print(f"  LoRA LR         : {args.lora_lr:.1e}   DepthMLP LR: {args.depth_mlp_lr:.1e}")
    print(f"  GPU             : cuda:{args.gpu}")
    print(f"  Checkpoint dir  : {ckpt_dir}")
    print("=" * 72 + "\n")

    # ── Load InternVL ─────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_path, args.gpu)
    model            = attach_lora(model)
    img_ctx_id       = model.img_context_token_id
    bos_offset       = 1 if tokenizer.bos_token_id is not None else 0
    print(f"bos_token_id={tokenizer.bos_token_id}  bos_offset={bos_offset}")

    # ── Load DINOv2 (second encoder) ─────────────────────────────────────────
    dino_model, dino_proc = load_dino(args.dino_model, device)

    # ── Build InternViT depth injector (hooks inside model() call) ────────────
    vit_injector = InternViTDepthInjector(
        vision_model      = model.vision_model,
        vit_depth_layers  = VIT_DEPTH_LAYERS,
        llm_inject_blocks = LLM_INJECT_BLOCKS,
        vit_hidden        = VIT_HIDDEN,
        llm_hidden        = LLM_HIDDEN,
        n_frames          = N_FRAMES,
    ).to(device)
    vit_injector.register_llm_hooks(model.language_model, bos_offset=bos_offset)

    # ── Build DINOv2 injector (separate pre-pass, hooks at LLM [4,8,12]) ─────
    dino_injector = DINOv2Injector(
        dino_model    = dino_model,
        dino_processor = dino_proc,
        vit_layers    = DINO_DEPTH_LAYERS,
        llm_blocks    = DINO_LLM_BLOCKS,
        dino_hidden   = DINO_HIDDEN,
        llm_hidden    = LLM_HIDDEN,
        n_frames      = N_FRAMES,
        img_size      = DINO_IMG_SIZE,
        ctx_per_frame = DINO_CTX_PER_FRAME,
    ).to(device)
    dino_injector.register_llm_hooks(model.language_model)

    head = MultiTaskHead(hidden=LLM_HIDDEN,
                         n_step=len(STEP_LABELS), n_stage=len(STAGE_LABELS)).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── Dataset statistics ────────────────────────────────────────────────────
    dataset_stats(data_dir)

    train_ds = SurgicalDataset(data_dir / "train.json", tokenizer, img_ctx_id, args.frames_base)
    val_ds   = SurgicalDataset(data_dir / "val.json",   tokenizer, img_ctx_id, args.frames_base)
    test_ds  = SurgicalDataset(data_dir / "test.json",  tokenizer, img_ctx_id, args.frames_base)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    cfn          = lambda b: collate_fn(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, persistent_workers=True, collate_fn=cfn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, persistent_workers=True, collate_fn=cfn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, persistent_workers=True, collate_fn=cfn)

    # ── Class weights (inverse-frequency) ────────────────────────────────────
    step_counts = torch.zeros(len(STEP_LABELS))
    for s in train_ds.samples:
        if s["main_tag"] == "step_classification":
            step_counts[LABEL2ID["step_classification"][s["meta"]["step_id"]]] += 1
    step_counts   = step_counts.clamp(min=1)
    step_weights  = (1.0 / step_counts)
    step_weights  = (step_weights / step_weights.mean()).to(device)

    # ── Optimizer: four param groups ─────────────────────────────────────────
    lora_params      = [p for p in model.language_model.parameters() if p.requires_grad]
    head_params      = list(head.parameters())
    vit_mlp_params   = list(vit_injector.depth_mlps.parameters())
    dino_mlp_params  = list(dino_injector.depth_mlps.parameters())
    wd = cfg["training"]["weight_decay"]

    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    n_vit  = sum(p.numel() for p in vit_mlp_params)
    n_dino = sum(p.numel() for p in dino_mlp_params)
    print(f"Trainable: LoRA={n_lora:,}  head={n_head:,}  "
          f"vit_mlps={n_vit:,}  dino_mlps={n_dino:,}  "
          f"total={n_lora+n_head+n_vit+n_dino:,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params,     "lr": args.lora_lr,      "weight_decay": wd},
        {"params": head_params,     "lr": args.lora_lr,      "weight_decay": wd},
        {"params": vit_mlp_params,  "lr": args.depth_mlp_lr, "weight_decay": wd},
        {"params": dino_mlp_params, "lr": args.depth_mlp_lr, "weight_decay": wd},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lora_lr / 20
    )

    ACCUM = 8
    best  = {"score": -1.0, "epoch": 0,
             "head": None, "lora": None, "vit_depth": None, "dino_depth": None}
    patience_left = args.patience

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); head.train(); vit_injector.train(); dino_injector.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            logits, tags, y = forward_pooled_depth(
                batch, model, head, device, vit_injector, dino_model, dino_injector)

            if tags[0] == "step_classification":
                loss = F.cross_entropy(logits["step_logits"],  y, weight=step_weights)
            else:
                loss = F.cross_entropy(logits["stage_logits"], y)

            (loss / ACCUM).backward()
            if i % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_params + head_params + vit_mlp_params + dino_mlp_params,
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())

        train_loss = running / len(train_ds)
        scheduler.step()
        step_res, stage_res, *_ = evaluate(
            model, head, val_loader, device, vit_injector,
            dino_model, dino_injector, desc=f"Val {epoch}")

        score = step_res["macro_f1"]
        dt    = time.time() - t0
        print(f"\nEpoch {epoch:02d} | {dt/60:.1f}min | lr={scheduler.get_last_lr()[0]:.2e} | "
              f"loss={train_loss:.4f} | "
              f"step acc={step_res['acc']:.3f} f1={step_res['macro_f1']:.3f} | "
              f"stage acc={stage_res['acc']:.3f} f1={stage_res['macro_f1']:.3f} | "
              f"score={score:.3f}")

        if score > best["score"]:
            best.update({
                "score":      score, "epoch": epoch,
                "head":       copy.deepcopy(head.state_dict()),
                "lora":       copy.deepcopy(model.language_model.state_dict()),
                "vit_depth":  copy.deepcopy(vit_injector.depth_mlps.state_dict()),
                "dino_depth": copy.deepcopy(dino_injector.depth_mlps.state_dict()),
            })
            patience_left = args.patience
            torch.save({
                "epoch":             epoch,
                "score":             score,
                "head":              best["head"],
                "lora":              best["lora"],
                "vit_depth_mlps":    best["vit_depth"],
                "dino_depth_mlps":   best["dino_depth"],
                "vit_depth_layers":  VIT_DEPTH_LAYERS,
                "llm_inject_blocks": LLM_INJECT_BLOCKS,
                "dino_depth_layers": DINO_DEPTH_LAYERS,
                "dino_llm_blocks":   DINO_LLM_BLOCKS,
                "bos_offset":        bos_offset,
            }, ckpt_dir / "best_vlm_depth.pth")
            print(f"  ↑ new best (score={score:.3f}) — checkpoint saved")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ── Test with best checkpoint ─────────────────────────────────────────────
    print(f"\n=== Best epoch: {best['epoch']}  score: {best['score']:.4f} ===")
    head.load_state_dict(best["head"])
    model.language_model.load_state_dict(best["lora"], strict=False)
    vit_injector.depth_mlps.load_state_dict(best["vit_depth"])
    dino_injector.depth_mlps.load_state_dict(best["dino_depth"])

    test_step, test_stage, ys_step, ps_step, ys_stage, ps_stage = evaluate(
        model, head, test_loader, device, vit_injector,
        dino_model, dino_injector, desc="Test")

    print(f"\nTEST step : {test_step}")
    print(f"TEST stage: {test_stage}")

    # ── Build integer→label look-ups ─────────────────────────────────────────
    id2step  = {v: k for k, v in LABEL2ID["step_classification"].items()}
    id2stage = {v: k for k, v in LABEL2ID["stage_classification"].items()}

    # ── Full per-class reports + confusion matrices ───────────────────────────
    reports_dir = ckpt_dir / "reports"
    step_report  = full_report(ys_step,  ps_step,  id2step,  "Step Classification",
                               out_dir=reports_dir, exclude_class=0)
    stage_report = full_report(ys_stage, ps_stage, id2stage, "Stage Classification",
                               out_dir=reports_dir)

    # ── Save complete test results ────────────────────────────────────────────
    with open(ckpt_dir / "test_results_vlm_depth.json", "w") as f:
        json.dump({
            "best_epoch":        best["epoch"],
            "step_summary":      test_step,
            "stage_summary":     test_stage,
            "step_full_report":  step_report,
            "stage_full_report": stage_report,
            "vit_depth_layers":  VIT_DEPTH_LAYERS,
            "llm_inject_blocks": LLM_INJECT_BLOCKS,
            "dino_depth_layers": DINO_DEPTH_LAYERS,
            "dino_llm_blocks":   DINO_LLM_BLOCKS,
            "lora_lr":           args.lora_lr,
            "depth_mlp_lr":      args.depth_mlp_lr,
        }, f, indent=2)
    print(f"\nFull test results saved to: {ckpt_dir / 'test_results_vlm_depth.json'}")


if __name__ == "__main__":
    main()

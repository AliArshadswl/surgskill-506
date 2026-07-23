"""
train_vlm_ablation.py — Ablation study for dual-encoder depth injection (InternVL3.5-1B).

Runs the following ablations one-by-one, controlled via --ablation flag:

  1. lora_only         : InternVL + LoRA, NO depth injection of any kind
  2. internvit_only    : InternViT injection only (blocks [1,2,3]), DINOv2 disabled
  3. dino_only         : DINOv2 injection only (blocks [4,8,12]), InternViT disabled
  4. both              : Both encoders (reproduce your existing results)
  5. internvit_at_dino : InternViT features injected at DINOv2 blocks [4,8,12] instead of [1,2,3]
  6. dino_at_internvit : DINOv2 features injected at InternViT blocks [1,2,3] instead of [4,8,12]
  7. swapped_blocks    : Both encoders but blocks fully swapped (InternViT→[4,8,12], DINOv2→[1,2,3])
  8. random_init       : Both encoders but DepthMLPs use random (non-near-zero) init
  9. zero_init         : Both encoders but DepthMLPs use exactly zero init (control for near-zero)

Usage:
  # Run a single ablation
  python train_vlm_ablation.py --ablation lora_only    --config config.yaml --dino_model /path/to/dino
  python train_vlm_ablation.py --ablation internvit_only ...
  python train_vlm_ablation.py --ablation dino_only    ...
  python train_vlm_ablation.py --ablation swapped_blocks ...
  python train_vlm_ablation.py --ablation random_init  ...

  # Run ALL ablations sequentially (one process per ablation)
  for abl in lora_only internvit_only dino_only internvit_at_dino dino_at_internvit swapped_blocks random_init zero_init; do
      python train_vlm_ablation.py --ablation $abl --config config.yaml --dino_model /path/to/dino --gpu 1
  done

Results are saved per-ablation under:
  <checkpoint_dir>/ablations/<ablation_name>/
    best_<ablation_name>.pth
    test_results_<ablation_name>.json
    reports/confusion_matrix_*.csv
    reports/report_*.json

A combined summary is appended to:
  <checkpoint_dir>/ablations/ablation_summary.json
"""

import copy
import json
import os
import random
import time
import types
from pathlib import Path

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


# ── Constants ──────────────────────────────────────────────────────────────────

N_FRAMES      = 8
CTX_PER_IMAGE = 256
IMG_SIZE      = 448
LLM_HIDDEN    = 1024
VIT_HIDDEN    = 1024

VIT_DEPTH_LAYERS  = [7, 15, 23]
LLM_INJECT_BLOCKS = [1, 2, 3]

STEP_LABELS  = [f"step_{i:03d}" for i in range(0, 14)]
STAGE_LABELS = ["stage_01", "stage_02", "stage_03"]

LABEL2ID = {
    "step_classification":  {lb: i for i, lb in enumerate(STEP_LABELS)},
    "stage_classification": {lb: i for i, lb in enumerate(STAGE_LABELS)},
}

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

DINO_HIDDEN        = 1024
DINO_IMG_SIZE      = 518
DINO_DEPTH_LAYERS  = [6, 12, 18]
DINO_LLM_BLOCKS    = [4, 8, 12]
DINO_CTX_PER_FRAME = CTX_PER_IMAGE

ALLOWED_TAGS = {"step_classification", "stage_classification"}

# ── All ablation configurations ────────────────────────────────────────────────

ABLATION_CONFIGS = {
    # ------------------------------------------------------------------
    # 1. Baseline: LoRA only, no injection whatsoever
    # ------------------------------------------------------------------
    "lora_only": {
        "use_internvit": False,
        "use_dino":      False,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "LoRA fine-tune only, zero depth injection",
    },
    # ------------------------------------------------------------------
    # 2. InternViT injection only
    # ------------------------------------------------------------------
    "internvit_only": {
        "use_internvit": True,
        "use_dino":      False,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "InternViT intermediate features → LLM blocks [1,2,3]",
    },
    # ------------------------------------------------------------------
    # 3. DINOv2 injection only
    # ------------------------------------------------------------------
    "dino_only": {
        "use_internvit": False,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "DINOv2 intermediate features → LLM blocks [4,8,12]",
    },
    # ------------------------------------------------------------------
    # 4. Both encoders — reproduce your published results
    # ------------------------------------------------------------------
    "both": {
        "use_internvit": True,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "Both encoders, original block assignments (reproduces reported results)",
    },
    # ------------------------------------------------------------------
    # 5. InternViT injected at DINOv2 positions [4,8,12]
    # ------------------------------------------------------------------
    "internvit_at_dino_blocks": {
        "use_internvit": True,
        "use_dino":      False,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       DINO_LLM_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "InternViT only, injected at blocks [4,8,12] (DINOv2 positions)",
    },
    # ------------------------------------------------------------------
    # 6. DINOv2 injected at InternViT positions [1,2,3]
    # ------------------------------------------------------------------
    "dino_at_internvit_blocks": {
        "use_internvit": False,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      LLM_INJECT_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "DINOv2 only, injected at blocks [1,2,3] (InternViT positions)",
    },
    # ------------------------------------------------------------------
    # 7. Both encoders with fully swapped block assignments
    # ------------------------------------------------------------------
    "swapped_blocks": {
        "use_internvit": True,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       DINO_LLM_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      LLM_INJECT_BLOCKS,
        "depth_mlp_init": "near_zero",
        "description": "Both encoders, block assignments fully swapped",
    },
    # ------------------------------------------------------------------
    # 8. Standard random init (std=0.02)
    # ------------------------------------------------------------------
    "random_init": {
        "use_internvit": True,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "random",
        "description": "Both encoders, DepthMLPs with standard random init (std=0.02)",
    },
    # ------------------------------------------------------------------
    # 9. Exactly zero output layer
    # ------------------------------------------------------------------
    "zero_init": {
        "use_internvit": True,
        "use_dino":      True,
        "vit_blocks":    VIT_DEPTH_LAYERS,
        "vit_llm":       LLM_INJECT_BLOCKS,
        "dino_blocks":   DINO_DEPTH_LAYERS,
        "dino_llm":      DINO_LLM_BLOCKS,
        "depth_mlp_init": "zero",
        "description": "Both encoders, DepthMLPs with exactly-zero output init",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Full per-class report + confusion matrix ───────────────────────────────────

def full_report(y_true, y_pred, int_to_label, task_name,
                out_dir=None, exclude_class=None):
    if not y_true:
        print(f"[{task_name}] No predictions collected.")
        return {}
    labels_present = sorted(set(y_true) | set(y_pred))
    if exclude_class is not None:
        labels_present = [l for l in labels_present if l != exclude_class]
    names = [int_to_label.get(i, str(i)) for i in labels_present]
    print(f"\n{'─'*72}")
    print(f"  {task_name.upper()}")
    print(f"{'─'*72}")
    print(classification_report(y_true, y_pred, labels=labels_present,
                                target_names=names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    report_dict = classification_report(
        y_true, y_pred, labels=labels_present, target_names=names,
        output_dict=True, zero_division=0)
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
        import csv
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = task_name.replace(" ", "_").lower()
        with open(out_dir / f"confusion_matrix_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for i, row in enumerate(cm):
                w.writerow([names[i]] + list(row))
        with open(out_dir / f"report_{tag}.json", "w") as f:
            json.dump(result, f, indent=2)
    return result


# ── DepthMLP with configurable init ───────────────────────────────────────────

class DepthMLP(nn.Module):
    """
    Projects ViT intermediate features into LLM hidden space.
    init_mode controls the output layer weight initialisation:
      'near_zero' : std=1e-3  (original — smooth warm-up from baseline)
      'random'    : std=0.02  (standard Kaiming/Xavier-like scale)
      'zero'      : all zeros (exactly zero start, still learnable)
    """
    def __init__(self, vit_hidden=VIT_HIDDEN, llm_hidden=LLM_HIDDEN,
                 init_mode="near_zero"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)

        if init_mode == "near_zero":
            nn.init.normal_(self.net[2].weight, std=1e-3)
        elif init_mode == "random":
            nn.init.normal_(self.net[2].weight, std=0.02)
        elif init_mode == "zero":
            nn.init.zeros_(self.net[2].weight)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x.to(self.net[0].weight.dtype))


# ── DINOv2 loader ──────────────────────────────────────────────────────────────

def load_dino(dino_path, device):
    print(f"[DINOv2] Loading from {dino_path}")
    proc = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    dino = AutoModel.from_pretrained(
        dino_path, local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device).eval()
    dino.requires_grad_(False)
    n = sum(p.numel() for p in dino.parameters())
    print(f"[DINOv2] Ready: hidden={dino.config.hidden_size}  "
          f"layers={dino.config.num_hidden_layers}  params={n/1e6:.1f}M")
    return dino, proc


# ── DINOv2 Injector ────────────────────────────────────────────────────────────

class DINOv2Injector(nn.Module):
    def __init__(self, dino_model, dino_processor,
                 vit_layers=DINO_DEPTH_LAYERS,
                 llm_blocks=DINO_LLM_BLOCKS,
                 dino_hidden=DINO_HIDDEN,
                 llm_hidden=LLM_HIDDEN,
                 n_frames=N_FRAMES,
                 img_size=DINO_IMG_SIZE,
                 ctx_per_frame=DINO_CTX_PER_FRAME,
                 init_mode="near_zero"):
        super().__init__()
        assert len(vit_layers) == len(llm_blocks)
        self._dino_processor = dino_processor
        self.vit_layers    = sorted(vit_layers)
        self.llm_blocks    = llm_blocks
        self.n_frames      = n_frames
        self.img_size      = img_size
        self.ctx_per_frame = ctx_per_frame
        self.n_visual      = n_frames * ctx_per_frame
        self.depth_mlps    = nn.ModuleList([
            DepthMLP(dino_hidden, llm_hidden, init_mode=init_mode)
            for _ in range(len(vit_layers))
        ])
        self._dino_blocks  = list(dino_model.encoder.layer)
        self._vit_feats:  dict = {}
        self._vit_hooks:  list = []
        self._llm_hooks:  list = []
        self._injections: dict = {}
        for p in dino_model.parameters():
            p.requires_grad = False
        dino_model.eval()
        print(f"[DINOv2Injector] ViT layers {self.vit_layers} → LLM blocks {self.llm_blocks}  init={init_mode}")

    def _attach_vit_hooks(self, dino_model):
        self._detach_vit_hooks()
        self._vit_feats.clear()
        for layer_idx in self.vit_layers:
            def _hook(module, inp, out, _idx=layer_idx):
                h = out[0] if isinstance(out, tuple) else out
                self._vit_feats[_idx] = h[:, 1:, :].detach()
            self._vit_hooks.append(
                self._dino_blocks[layer_idx].register_forward_hook(_hook))

    def _detach_vit_hooks(self):
        for h in self._vit_hooks: h.remove()
        self._vit_hooks.clear()

    def register_llm_hooks(self, language_model):
        for h in self._llm_hooks: h.remove()
        self._llm_hooks.clear()
        raw = language_model
        visited = set()
        while hasattr(raw, "base_model"):
            if id(raw) in visited: break
            visited.add(id(raw)); raw = raw.base_model
        if hasattr(raw, "model") and not hasattr(raw, "layers"):
            raw = raw.model
        if not hasattr(raw, "layers"):
            raise RuntimeError(f"[DINOv2Injector] Cannot find .layers on LLM.")
        blocks = raw.layers
        print(f"[DINOv2Injector] Registering hooks at LLM blocks {self.llm_blocks}")
        for depth_idx, block_idx in enumerate(self.llm_blocks):
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
            self._llm_hooks.append(blocks[block_idx].register_forward_hook(_hook))

    @torch.no_grad()
    def prepare_injections(self, dino_model, image_paths, B):
        self._vit_feats.clear(); self._injections.clear()
        self._attach_vit_hooks(dino_model)
        images = []
        for p in image_paths:
            try:   images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (self.img_size, self.img_size), (128,128,128)))
        dev = next(dino_model.parameters()).device
        # Bypass processor's numpy path — preprocess without touching numpy
        _resize = transforms.Resize((self.img_size, self.img_size))
        _norm   = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        pixel_values = torch.stack([
            _norm(_pil_to_float_tensor(_resize(img))) for img in images
        ]).to(dev)
        dino_model(pixel_values=pixel_values)
        self._detach_vit_hooks()
        sqrt_ctx = int(round(self.ctx_per_frame ** 0.5))
        for depth_idx, layer_idx in enumerate(self.vit_layers):
            feats = self._vit_feats.get(layer_idx)
            if feats is None: continue
            BN_f, n_patches, D = feats.shape
            sqrt_p  = int(round(n_patches ** 0.5))
            spatial = feats.permute(0, 2, 1).reshape(BN_f, D, sqrt_p, sqrt_p)
            pooled  = F.adaptive_avg_pool2d(spatial.float(), output_size=(sqrt_ctx, sqrt_ctx))
            flat    = pooled.reshape(BN_f, D, self.ctx_per_frame).permute(0, 2, 1).to(feats.dtype)
            combined  = flat.reshape(B, self.n_visual, D)
            projected = self.depth_mlps[depth_idx](combined)
            self._injections[depth_idx] = projected

    def clear_state(self):
        self._vit_feats.clear(); self._injections.clear()


# ── InternViT Depth Injector ───────────────────────────────────────────────────

class InternViTDepthInjector(nn.Module):
    def __init__(self, vision_model,
                 vit_depth_layers=VIT_DEPTH_LAYERS,
                 llm_inject_blocks=LLM_INJECT_BLOCKS,
                 vit_hidden=VIT_HIDDEN,
                 llm_hidden=LLM_HIDDEN,
                 n_frames=N_FRAMES,
                 init_mode="near_zero"):
        super().__init__()
        assert len(vit_depth_layers) == len(llm_inject_blocks)
        self.vit_depth_layers  = sorted(vit_depth_layers)
        self.llm_inject_blocks = llm_inject_blocks
        self.vit_hidden        = vit_hidden
        self.llm_hidden        = llm_hidden
        self.n_frames          = n_frames
        self.n_visual          = n_frames * CTX_PER_IMAGE
        self.depth_mlps        = nn.ModuleList([
            DepthMLP(vit_hidden, llm_hidden, init_mode=init_mode)
            for _ in range(len(vit_depth_layers))
        ])
        self._vit_feats:  dict = {}
        self._injections: dict = {}
        self._vit_hooks:  list = []
        self._llm_hooks:  list = []
        self._encoder_blocks = self._find_encoder_blocks(vision_model)
        print(f"[InternViTInjector] ViT layers {self.vit_depth_layers} → "
              f"LLM blocks {self.llm_inject_blocks}  init={init_mode}")

    @staticmethod
    def _find_encoder_blocks(vision_model):
        for path in [("encoder", "layers"), ("encoder", "layer")]:
            obj = vision_model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None: break
            if obj is not None and hasattr(obj, "__len__"):
                return obj
        for name, mod in vision_model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 4:
                return mod
        raise RuntimeError("[InternViTInjector] Cannot find encoder blocks.")

    def attach_vit_hooks(self):
        self.detach_vit_hooks()
        self._vit_feats.clear()
        for layer_idx in self.vit_depth_layers:
            def _hook(module, inp, out, _idx=layer_idx):
                hidden = out[0] if isinstance(out, tuple) else out
                self._vit_feats[_idx] = hidden.detach()
            self._vit_hooks.append(
                self._encoder_blocks[layer_idx].register_forward_hook(_hook))

    def detach_vit_hooks(self):
        for h in self._vit_hooks: h.remove()
        self._vit_hooks.clear()

    def register_llm_hooks(self, language_model, bos_offset=0):
        for h in self._llm_hooks: h.remove()
        self._llm_hooks.clear()
        raw = language_model
        visited = set()
        while hasattr(raw, "base_model"):
            if id(raw) in visited: break
            visited.add(id(raw)); raw = raw.base_model
        if hasattr(raw, "model") and not hasattr(raw, "layers"):
            raw = raw.model
        if not hasattr(raw, "layers"):
            raise RuntimeError("[InternViTInjector] Cannot find .layers on LLM.")
        blocks = raw.layers
        print(f"[InternViTInjector] Qwen3: {len(blocks)} blocks, bos_offset={bos_offset}")

        def _prepare_hook(module, inp, out):
            self._prepare_injections()
            return out
        self._llm_hooks.append(blocks[0].register_forward_hook(_prepare_hook))

        _start = bos_offset
        _n_vis = self.n_visual
        for depth_idx, block_idx in enumerate(self.llm_inject_blocks):
            def _inject_hook(module, inp, out, _d=depth_idx, _s=_start, _n=_n_vis):
                payload = self._injections.get(_d)
                if payload is None: return out
                hidden = out[0] if isinstance(out, tuple) else out
                B, T, H = hidden.shape
                inj = payload.to(dtype=hidden.dtype, device=hidden.device)
                add_len = min(_n, T - _s)
                if add_len <= 0: return out
                updated = hidden.clone()
                updated[:, _s:_s + add_len, :] = (
                    hidden[:, _s:_s + add_len, :] + inj[:, :add_len, :])
                return (updated,) + out[1:] if isinstance(out, tuple) else updated
            self._llm_hooks.append(blocks[block_idx].register_forward_hook(_inject_hook))
        print(f"[InternViTInjector] Hooks: prepare@block0, inject@blocks {self.llm_inject_blocks}")

    def _prepare_injections(self):
        self._injections.clear()
        if not self._vit_feats: return
        first = next(iter(self._vit_feats.values()))
        BN = first.shape[0]
        B  = BN // self.n_frames
        if B <= 0 or B * self.n_frames != BN: return
        for depth_idx, layer_idx in enumerate(self.vit_depth_layers):
            feats = self._vit_feats.get(layer_idx)
            if feats is None: continue
            BN, n_tokens, D = feats.shape
            sqrt_p = int(round(n_tokens ** 0.5))
            if sqrt_p * sqrt_p != n_tokens:
                feats    = feats[:, 1:, :]
                n_tokens = feats.shape[1]
                sqrt_p   = int(round(n_tokens ** 0.5))
            spatial  = feats.permute(0, 2, 1).reshape(BN, D, sqrt_p, sqrt_p)
            pooled   = F.adaptive_avg_pool2d(spatial.float(), output_size=(16, 16))
            flat     = pooled.reshape(BN, D, CTX_PER_IMAGE).permute(0, 2, 1).to(feats.dtype)
            combined = flat.reshape(B, self.n_visual, D)
            self._injections[depth_idx] = self.depth_mlps[depth_idx](combined)

    def clear_state(self):
        self._vit_feats.clear(); self._injections.clear()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path, gpu_id):
    device_str = f"cuda:{gpu_id}"
    tokenizer  = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    import transformers.modeling_utils as _mu
    _orig = getattr(_mu, "get_total_byte_count", None)
    if _orig is not None:
        def _patched(model, device_map, hf_quantizer=None):
            if not hasattr(model, "all_tied_weights_keys"):
                object.__setattr__(model, "all_tied_weights_keys", {})
            return _orig(model, device_map, hf_quantizer)
        _mu.get_total_byte_count = _patched

    # FIX: torch_dtype= (not dtype=)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device_str)

    _pe = model.vision_model.embeddings.patch_embedding
    def _fp32_conv(self, x):
        orig = x.dtype
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            out = torch.nn.functional.conv2d(
                x.to(torch.float32), self.weight.to(torch.float32),
                self.bias.to(torch.float32) if self.bias is not None else None,
                self.stride, self.padding, self.dilation, self.groups)
        finally:
            torch.backends.cudnn.enabled = prev
        return out.to(orig)
    _pe.forward = types.MethodType(_fp32_conv, _pe)

    img_ctx_id = None
    for tok in ["<IMG_CONTEXT>", "<img_context>", "<image_context>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            img_ctx_id = tid; break
    if img_ctx_id is None:
        img_ctx_id = 151671
    model.img_context_token_id = img_ctx_id
    print(f"Loaded {type(model).__name__}  img_ctx_id={img_ctx_id}")
    return model, tokenizer


def attach_lora(model):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()
    model.vision_model.requires_grad_(False)
    if hasattr(model, "mlp1"):
        model.mlp1.requires_grad_(False)
    if not hasattr(model.language_model, "_orig_prepare_inputs_for_generation"):
        model.language_model._orig_prepare_inputs_for_generation = \
            model.language_model.prepare_inputs_for_generation
    def safe_prepare(self, input_ids, past_key_values=None,
                     attention_mask=None, inputs_embeds=None, **kwargs):
        bad = False
        try:
            if past_key_values is not None:
                if len(past_key_values) == 0: bad = True
                elif past_key_values[0] is None: bad = True
                elif isinstance(past_key_values[0], (tuple,list)) and past_key_values[0][0] is None:
                    bad = True
        except Exception: bad = True
        if bad: past_key_values = None
        return self._orig_prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    model.language_model.prepare_inputs_for_generation = types.MethodType(
        safe_prepare, model.language_model)
    return model


# ── Classification head ────────────────────────────────────────────────────────

class MultiTaskHead(nn.Module):
    def __init__(self, hidden=LLM_HIDDEN, n_step=14, n_stage=3):
        super().__init__()
        self.step  = nn.Linear(hidden, n_step)
        self.stage = nn.Linear(hidden, n_stage)
    def forward(self, x):
        return {"step_logits": self.step(x), "stage_logits": self.stage(x)}


# ── Dataset ────────────────────────────────────────────────────────────────────

def _pil_to_float_tensor(img):
    """Convert PIL image to float tensor without touching numpy (avoids dual-numpy conflict)."""
    w, h = img.size
    c = len(img.getbands())
    buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return buf.reshape(h, w, c).permute(2, 0, 1).float() / 255.0

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(_pil_to_float_tensor),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

def build_inputs(tokenizer, n_frames, tag, img_ctx_id):
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    n_ctx      = n_frames * CTX_PER_IMAGE
    prompt     = "Classify step." if tag == "step_classification" else "Classify stage."
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ids = []
    if bos is not None: ids.append(bos)
    ids += [img_ctx_id] * n_ctx + prompt_ids
    if eos is not None: ids.append(eos)
    input_ids      = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    image_flags    = torch.ones((n_frames,), dtype=torch.long)
    return input_ids, attention_mask, image_flags

class SurgicalDataset(Dataset):
    def __init__(self, json_path, tokenizer, img_ctx_id, frames_base=""):
        with open(json_path, encoding="utf-8") as f:
            all_samples = json.load(f)
        self.samples     = [s for s in all_samples if s.get("main_tag") in ALLOWED_TAGS]
        self.tokenizer   = tokenizer
        self.img_ctx_id  = img_ctx_id
        self.frames_base = frames_base

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        tag = s["main_tag"]
        meta = s.get("meta", {}) or {}
        label_str = meta["step_id"] if tag == "step_classification" else meta["stage_id"]
        y         = LABEL2ID[tag][label_str]
        frames    = s["frames"][:N_FRAMES]
        while len(frames) < N_FRAMES: frames.append(frames[-1])
        if self.frames_base:
            frames = [p if os.path.isabs(p) else os.path.join(self.frames_base, p)
                      for p in frames]
        # FIX: convert PIL image to numpy explicitly before ToTensor
        # to avoid the numpy version conflict in DataLoader workers
        pixel_values = torch.stack([
            img_transform(Image.open(p).convert("RGB"))
            for p in frames
        ])
        input_ids, attention_mask, image_flags = build_inputs(
            self.tokenizer, N_FRAMES, tag, self.img_ctx_id)
        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "attention_mask": attention_mask, "image_flags": image_flags,
                "frames": frames, "tag": tag, "y": y}

def collate_fn(batch, pad_id=0):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    def pad(x, v):
        if x.shape[0] == max_len: return x
        return torch.cat([x, torch.full((max_len - x.shape[0],), v, dtype=x.dtype)])
    return {
        "pixel_values":   torch.stack([b["pixel_values"]               for b in batch]),
        "input_ids":      torch.stack([pad(b["input_ids"],      pad_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0)      for b in batch]),
        "image_flags":    torch.stack([b["image_flags"]                 for b in batch]),
        "frames":         [p for b in batch for p in b["frames"]],
        "tags":           [b["tag"] for b in batch],
        "y":              torch.tensor([b["y"] for b in batch], dtype=torch.long),
    }


# ── Forward pass ───────────────────────────────────────────────────────────────

def forward_pooled(batch, model, head, device,
                   vit_injector=None, dino_model=None, dino_injector=None):
    """
    Unified forward for all ablations.
    vit_injector=None  → no InternViT injection (lora_only, dino_only)
    dino_injector=None → no DINOv2 injection   (lora_only, internvit_only)
    """
    B, N, C, H, W = batch["pixel_values"].shape

    # DINOv2 pre-pass (only when active)
    if dino_model is not None and dino_injector is not None:
        dino_injector.prepare_injections(dino_model, batch["frames"], B)

    pv             = batch["pixel_values"].to(device=device, dtype=torch.bfloat16).view(B*N, C, H, W)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    image_flags    = batch["image_flags"].to(device).view(B*N)

    if vit_injector is not None:
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

    if vit_injector is not None:
        vit_injector.detach_vit_hooks()

    seq_lens = attention_mask.sum(dim=1) - 1
    pooled   = out.hidden_states[-1][
        torch.arange(B, device=device), seq_lens, :].float()
    logits   = head(pooled)

    if vit_injector is not None:  vit_injector.clear_state()
    if dino_injector is not None: dino_injector.clear_state()
    return logits, batch["tags"], batch["y"].to(device)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, head, loader, device,
             vit_injector=None, dino_model=None, dino_injector=None,
             desc="eval"):
    model.eval(); head.eval()
    if vit_injector is not None:  vit_injector.eval()
    if dino_injector is not None: dino_injector.eval()
    ys_step, ps_step, ys_stage, ps_stage = [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        logits, tags, y = forward_pooled(
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
    return step_res, stage_res, ys_step, ps_step, ys_stage, ps_stage


# ── Single ablation run ────────────────────────────────────────────────────────

def run_ablation(ablation_name, abl_cfg, args, data_dir, ckpt_dir,
                 dino_model, dino_proc):
    """
    Trains and evaluates one ablation configuration from scratch.
    Saves all outputs to ckpt_dir / ablation_name /.
    Returns a summary dict.
    """
    print("\n" + "=" * 72)
    print(f"  ABLATION: {ablation_name}")
    print(f"  {abl_cfg['description']}")
    print("=" * 72)

    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}")
    abl_dir = ckpt_dir / ablation_name
    abl_dir.mkdir(parents=True, exist_ok=True)

    cfg        = load_config(args.config)
    model_path = cfg["data"]["internvl_model"]

    # ── Load model fresh for every ablation ──────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_path, args.gpu)
    model            = attach_lora(model)
    img_ctx_id       = model.img_context_token_id
    bos_offset       = 1 if tokenizer.bos_token_id is not None else 0

    init_mode  = abl_cfg["depth_mlp_init"]

    # ── InternViT injector (None if not used) ────────────────────────────────
    vit_injector = None
    if abl_cfg["use_internvit"]:
        vit_injector = InternViTDepthInjector(
            vision_model      = model.vision_model,
            vit_depth_layers  = abl_cfg["vit_blocks"],
            llm_inject_blocks = abl_cfg["vit_llm"],
            vit_hidden        = VIT_HIDDEN,
            llm_hidden        = LLM_HIDDEN,
            n_frames          = N_FRAMES,
            init_mode         = init_mode,
        ).to(device)
        vit_injector.register_llm_hooks(model.language_model, bos_offset=bos_offset)

    # ── DINOv2 injector (None if not used) ────────────────────────────────────
    abl_dino_model    = dino_model    if abl_cfg["use_dino"] else None
    abl_dino_injector = None
    if abl_cfg["use_dino"]:
        abl_dino_injector = DINOv2Injector(
            dino_model     = dino_model,
            dino_processor = dino_proc,
            vit_layers     = abl_cfg["dino_blocks"],
            llm_blocks     = abl_cfg["dino_llm"],
            dino_hidden    = DINO_HIDDEN,
            llm_hidden     = LLM_HIDDEN,
            n_frames       = N_FRAMES,
            img_size       = DINO_IMG_SIZE,
            ctx_per_frame  = DINO_CTX_PER_FRAME,
            init_mode      = init_mode,
        ).to(device)
        abl_dino_injector.register_llm_hooks(model.language_model)

    head = MultiTaskHead(hidden=LLM_HIDDEN,
                         n_step=len(STEP_LABELS),
                         n_stage=len(STAGE_LABELS)).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cfn      = lambda b: collate_fn(b, pad_id)
    train_ds = SurgicalDataset(data_dir / "train.json", tokenizer, img_ctx_id, args.frames_base)
    val_ds   = SurgicalDataset(data_dir / "val.json",   tokenizer, img_ctx_id, args.frames_base)
    test_ds  = SurgicalDataset(data_dir / "test.json",  tokenizer, img_ctx_id, args.frames_base)

    # FIX: num_workers=0 to avoid numpy version conflict in worker subprocesses
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=0, collate_fn=cfn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=cfn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=cfn, pin_memory=True)

    # ── Class weights ─────────────────────────────────────────────────────────
    step_counts = torch.zeros(len(STEP_LABELS))
    for s in train_ds.samples:
        if s["main_tag"] == "step_classification":
            step_counts[LABEL2ID["step_classification"][s["meta"]["step_id"]]] += 1
    step_weights = (1.0 / step_counts.clamp(min=1))
    step_weights = (step_weights / step_weights.mean()).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for p in model.language_model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    extra_params = []
    if vit_injector is not None:
        extra_params += list(vit_injector.depth_mlps.parameters())
    if abl_dino_injector is not None:
        extra_params += list(abl_dino_injector.depth_mlps.parameters())

    wd = cfg["training"]["weight_decay"]
    param_groups = [
        {"params": lora_params,  "lr": args.lora_lr,      "weight_decay": wd},
        {"params": head_params,  "lr": args.lora_lr,      "weight_decay": wd},
    ]
    if extra_params:
        param_groups.append(
            {"params": extra_params, "lr": args.depth_mlp_lr, "weight_decay": wd})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lora_lr / 20)

    ACCUM = 8
    best  = {"score": -1.0, "epoch": 0,
             "head": None, "lora": None, "vit_depth": None, "dino_depth": None}
    patience_left = args.patience

    # Store per-epoch val metrics for convergence analysis
    epoch_log = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); head.train()
        if vit_injector is not None:      vit_injector.train()
        if abl_dino_injector is not None: abl_dino_injector.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, batch in enumerate(
            tqdm(train_loader, desc=f"[{ablation_name}] Epoch {epoch}", leave=False), start=1
        ):
            logits, tags, y = forward_pooled(
                batch, model, head, device,
                vit_injector, abl_dino_model, abl_dino_injector)
            if tags[0] == "step_classification":
                loss = F.cross_entropy(logits["step_logits"],  y, weight=step_weights)
            else:
                loss = F.cross_entropy(logits["stage_logits"], y)
            (loss / ACCUM).backward()
            if i % ACCUM == 0:
                all_trainable = lora_params + head_params + extra_params
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())

        train_loss = running / len(train_ds)
        scheduler.step()
        step_res, stage_res, *_ = evaluate(
            model, head, val_loader, device,
            vit_injector, abl_dino_model, abl_dino_injector,
            desc=f"[{ablation_name}] Val {epoch}")

        score = step_res["macro_f1"]
        dt    = time.time() - t0
        log_entry = {
            "epoch": epoch, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0],
            "step_acc": step_res["acc"], "step_macro_f1": step_res["macro_f1"],
            "step_kappa": step_res["kappa"],
            "stage_acc": stage_res["acc"], "stage_macro_f1": stage_res["macro_f1"],
            "stage_kappa": stage_res["kappa"],
        }
        epoch_log.append(log_entry)
        print(f"\n[{ablation_name}] Epoch {epoch:02d} | {dt/60:.1f}min | "
              f"loss={train_loss:.4f} | step_f1={step_res['macro_f1']:.3f} | "
              f"stage_f1={stage_res['macro_f1']:.3f}")

        if score > best["score"]:
            best.update({
                "score": score, "epoch": epoch,
                "head":  copy.deepcopy(head.state_dict()),
                "lora":  copy.deepcopy(model.language_model.state_dict()),
                "vit_depth":  copy.deepcopy(vit_injector.depth_mlps.state_dict())
                              if vit_injector is not None else None,
                "dino_depth": copy.deepcopy(abl_dino_injector.depth_mlps.state_dict())
                              if abl_dino_injector is not None else None,
            })
            patience_left = args.patience
            ckpt = {
                "epoch": epoch, "score": score,
                "ablation": ablation_name,
                "config": abl_cfg,
                "head": best["head"],
                "lora": best["lora"],
            }
            if best["vit_depth"]  is not None: ckpt["vit_depth_mlps"]  = best["vit_depth"]
            if best["dino_depth"] is not None: ckpt["dino_depth_mlps"] = best["dino_depth"]
            torch.save(ckpt, abl_dir / f"best_{ablation_name}.pth")
            print(f"  ↑ new best (score={score:.3f})")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"[{ablation_name}] Early stopping at epoch {epoch}.")
                break

    # ── Test with best checkpoint ──────────────────────────────────────────────
    print(f"\n[{ablation_name}] Best epoch: {best['epoch']}  score: {best['score']:.4f}")
    head.load_state_dict(best["head"])
    model.language_model.load_state_dict(best["lora"], strict=False)
    if vit_injector is not None and best["vit_depth"] is not None:
        vit_injector.depth_mlps.load_state_dict(best["vit_depth"])
    if abl_dino_injector is not None and best["dino_depth"] is not None:
        abl_dino_injector.depth_mlps.load_state_dict(best["dino_depth"])

    test_step, test_stage, ys_step, ps_step, ys_stage, ps_stage = evaluate(
        model, head, test_loader, device,
        vit_injector, abl_dino_model, abl_dino_injector,
        desc=f"[{ablation_name}] Test")

    print(f"[{ablation_name}] TEST step : {test_step}")
    print(f"[{ablation_name}] TEST stage: {test_stage}")

    id2step  = {v: k for k, v in LABEL2ID["step_classification"].items()}
    id2stage = {v: k for k, v in LABEL2ID["stage_classification"].items()}
    reports_dir = abl_dir / "reports"
    step_report  = full_report(ys_step,  ps_step,  id2step,  f"{ablation_name} Step",
                               out_dir=reports_dir, exclude_class=0)
    stage_report = full_report(ys_stage, ps_stage, id2stage, f"{ablation_name} Stage",
                               out_dir=reports_dir)

    result = {
        "ablation":          ablation_name,
        "description":       abl_cfg["description"],
        "use_internvit":     abl_cfg["use_internvit"],
        "use_dino":          abl_cfg["use_dino"],
        "vit_llm_blocks":    abl_cfg["vit_llm"],
        "dino_llm_blocks":   abl_cfg["dino_llm"],
        "depth_mlp_init":    abl_cfg["depth_mlp_init"],
        "best_epoch":        best["epoch"],
        "best_val_score":    best["score"],
        "step_summary":      test_step,
        "stage_summary":     test_stage,
        "step_full_report":  step_report,
        "stage_full_report": stage_report,
        "epoch_log":         epoch_log,
    }
    with open(abl_dir / f"test_results_{ablation_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Clean up GPU memory before next ablation
    del model, head, vit_injector, abl_dino_injector, optimizer, scheduler
    torch.cuda.empty_cache()

    return result


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary_table(all_results):
    """Print a compact comparison table across all ablations."""
    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Ablation':<28} {'Step Acc':>9} {'Step F1':>9} {'Step κ':>8} "
              f"{'Stg Acc':>9} {'Stg F1':>9} {'Stg κ':>8} {'BestEp':>7}")
    print(header)
    print("  " + "-" * 88)
    for r in all_results:
        ss  = r.get("step_summary",  {})
        sst = r.get("stage_summary", {})
        print(
            f"  {r['ablation']:<28} "
            f"{ss.get('acc', 0):.4f}    "
            f"{ss.get('macro_f1', 0):.4f}    "
            f"{ss.get('kappa', 0):.4f}   "
            f"{sst.get('acc', 0):.4f}    "
            f"{sst.get('macro_f1', 0):.4f}    "
            f"{sst.get('kappa', 0):.4f}   "
            f"{r.get('best_epoch', '?'):>5}"
        )
    print("=" * 90)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import warnings
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
    warnings.filterwarnings("ignore", message=".*use_reentrant.*")

    parser = argparse.ArgumentParser(description="VLM depth injection ablation study")
    parser.add_argument("--ablation", required=True,
                        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                        help="Which ablation to run. Pass 'all' to run every one sequentially.")
    parser.add_argument("--config",      default="/mnt/share/ali/surgical_assessment_v2/Retry/config.yaml")
    parser.add_argument("--data_dir",    default="/mnt/share/ali/surgical_assessment_v2/Retry")
    parser.add_argument("--frames_base", default="")
    parser.add_argument("--dino_model",  required=True,
                        help="Path to DINOv2 ViT-L/14 model directory")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--patience",    type=int,   default=7)
    parser.add_argument("--gpu",         type=int,   default=0)  # FIX: default=0 (safe default)
    parser.add_argument("--lora_lr",     type=float, default=2e-4)
    parser.add_argument("--depth_mlp_lr",type=float, default=1e-4)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"]  = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg      = load_config(args.config)
    device   = torch.device(f"cuda:{args.gpu}")
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / "ablations"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load DINOv2 once — shared (frozen) across all ablations
    dino_model, dino_proc = load_dino(args.dino_model, device)

    # Decide which ablations to run
    if args.ablation == "all":
        ablation_names = list(ABLATION_CONFIGS.keys())
    else:
        ablation_names = [args.ablation]

    print(f"\nRunning ablations: {ablation_names}")
    print(f"Checkpoint dir   : {ckpt_dir}\n")

    all_results = []

    # Load existing summary so partial runs can resume
    summary_path = ckpt_dir / "ablation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        completed = {r["ablation"] for r in existing}
        all_results = existing
        print(f"Found {len(completed)} completed ablations in summary: {completed}")
    else:
        completed = set()

    for name in ablation_names:
        if name in completed:
            print(f"\n[SKIP] {name} already in summary — delete summary to re-run.")
            continue
        abl_cfg = ABLATION_CONFIGS[name]
        result  = run_ablation(
            ablation_name = name,
            abl_cfg       = abl_cfg,
            args          = args,
            data_dir      = data_dir,
            ckpt_dir      = ckpt_dir,
            dino_model    = dino_model,
            dino_proc     = dino_proc,
        )
        all_results.append(result)
        # Save/update summary after every ablation so partial runs are preserved
        with open(summary_path, "w") as f:
            slim = []
            for r in all_results:
                slim.append({k: v for k, v in r.items()
                             if k not in ("step_full_report", "stage_full_report",
                                          "epoch_log")})
            json.dump(slim, f, indent=2)
        print(f"\n[Summary] Saved {len(all_results)} results → {summary_path}")

    print_summary_table(all_results)
    print(f"\nAll done. Full results in: {ckpt_dir}")


if __name__ == "__main__":
    main()
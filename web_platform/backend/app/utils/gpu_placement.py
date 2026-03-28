"""
Pick a torch device (cuda:i / cpu) for loading MedRAX tools.

Uses physical GPU indices from nvidia-smi and maps them to logical cuda:i
indices via CUDA_VISIBLE_DEVICES. Set CUDA_VISIBLE_DEVICES=2,1 so that
cuda:0 -> GPU 2 and cuda:1 -> GPU 1 (preferred order for "try 2, then 1").
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_physical_order(order: str) -> List[int]:
    out: List[int] = []
    for part in order.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def physical_to_logical_cuda() -> Dict[int, int]:
    """
    Map physical GPU index -> logical cuda index for this process.

    If CUDA_VISIBLE_DEVICES is unset, physical i maps to cuda:i.
    If set to e.g. "2,1", then physical 2 -> cuda:0, physical 1 -> cuda:1.
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        n = torch.cuda.device_count()
        return {i: i for i in range(n)}

    parts = [p.strip() for p in vis.split(",") if p.strip()]
    mapping: Dict[int, int] = {}
    for logical, token in enumerate(parts):
        if token.isdigit():
            mapping[int(token)] = logical
    return mapping


def query_free_memory_mib_by_physical() -> Dict[int, int]:
    """physical GPU index -> free memory (MiB) using nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.debug("nvidia-smi memory query failed: %s", e)
        return {}

    result: Dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            result[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return result


def _free_mib_torch_by_logical() -> Dict[int, int]:
    """Logical cuda index -> free MiB (fallback when nvidia-smi unavailable)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    out: Dict[int, int] = {}
    for i in range(torch.cuda.device_count()):
        try:
            free_b, _ = torch.cuda.mem_get_info(i)
            out[i] = free_b // (1024 * 1024)
        except Exception as e:
            logger.debug("torch.cuda.mem_get_info failed for cuda:%s: %s", i, e)
    return out


def select_tool_torch_device(_requires_gpu: bool, settings) -> str:
    """
    Resolve torch device string for tool construction.

    requires_gpu is informational (both branches may use GPU for optional-GPU tools).
    """
    if settings.FORCE_CPU:
        return "cpu"

    import torch

    if not torch.cuda.is_available():
        return "cpu"

    strategy = getattr(settings, "TOOL_GPU_STRATEGY", "auto") or "auto"

    if strategy == "fixed":
        fixed = getattr(settings, "TOOL_DEVICE", None) or ""
        if fixed and fixed != "auto":
            return fixed
        dev = getattr(settings, "DEVICE", "auto") or "auto"
        if dev and dev != "auto":
            return dev
        return "cuda:0"

    # auto: prefer physical order with free-memory threshold
    order = _parse_physical_order(getattr(settings, "TOOL_GPU_PHYSICAL_ORDER", "2,1") or "2,1")
    if not order:
        order = [2, 1]

    min_free = int(getattr(settings, "TOOL_GPU_MIN_FREE_MIB", 2048) or 2048)

    phys_map = physical_to_logical_cuda()
    free_phys = query_free_memory_mib_by_physical()

    if not free_phys:
        logical_free = _free_mib_torch_by_logical()
        for phys in order:
            if phys not in phys_map:
                continue
            logical = phys_map[phys]
            fm = logical_free.get(logical)
            if fm is not None and fm >= min_free:
                choice = f"cuda:{logical}"
                logger.info(
                    "TOOL_GPU auto (torch mem): picked %s (physical GPU %s, ~%s MiB free, threshold %s)",
                    choice,
                    phys,
                    fm,
                    min_free,
                )
                return choice
        best_logical: Optional[int] = None
        best_fm = -1
        for phys in order:
            if phys not in phys_map:
                continue
            logical = phys_map[phys]
            fm = logical_free.get(logical, 0)
            if fm > best_fm:
                best_fm = fm
                best_logical = logical
        if best_logical is not None:
            choice = f"cuda:{best_logical}"
            logger.warning(
                "TOOL_GPU auto (torch mem): no GPU met %s MiB; using %s (~%s MiB free)",
                min_free,
                choice,
                best_fm,
            )
            return choice
        if phys_map:
            first_logical = next(iter(phys_map.values()))
            return f"cuda:{first_logical}"
        return "cuda:0"

    # Prefer first GPU in order with enough free memory
    for phys in order:
        if phys not in phys_map:
            logger.debug("Physical GPU %s not visible to this process (CUDA_VISIBLE_DEVICES?)", phys)
            continue
        fm = free_phys.get(phys)
        if fm is None:
            continue
        if fm >= min_free:
            logical = phys_map[phys]
            choice = f"cuda:{logical}"
            logger.info(
                "TOOL_GPU auto: picked %s (physical GPU %s, %s MiB free >= %s MiB)",
                choice,
                phys,
                fm,
                min_free,
            )
            return choice

    # Nothing met threshold: choose visible candidate in order with most free memory
    best_phys: Optional[int] = None
    best_fm = -1
    for phys in order:
        if phys not in phys_map:
            continue
        fm = free_phys.get(phys, 0)
        if fm > best_fm:
            best_fm = fm
            best_phys = phys

    if best_phys is not None:
        logical = phys_map[best_phys]
        choice = f"cuda:{logical}"
        logger.warning(
            "TOOL_GPU auto: no GPU met %s MiB free; using %s (physical GPU %s, %s MiB free)",
            min_free,
            choice,
            best_phys,
            best_fm,
        )
        return choice

    return "cuda:0"

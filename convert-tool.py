#!/usr/bin/env python3
"""v2a-nyq: video-to-audio conversion with Nyquist-based composition and synthesis.

Python responsibilities:
- Decode video and convert frames to normalized HSV numeric features.
- Segment timeline and export numeric descriptors for Nyquist.
- Launch Nyquist engine for algorithmic mapping + audio synthesis.

Nyquist responsibilities (see v2a_nyq_engine.lsp / .sal):
- Tempo and scale decision (when not overridden).
- Harmony/melody/bass generation.
- Audio synthesis and WAV rendering.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from media_sync import probe_video_duration_seconds, probe_wav_duration_seconds, sync_wav_to_video_duration

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

from simple_kmeans import KMeans


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
INPUT_BATCH_RE = re.compile(r"^input-(\d+)\.(mp4|mov|mkv|avi|webm)$", re.IGNORECASE)
SCALES = ("auto", "major", "minor", "dorian", "mixolydian", "pentatonic")
MOODS = ("none", "cheerful", "epic")
C_M_PER_S = 299792458.0


@dataclass(frozen=True)
class SegmentFeature:
    hue_mean: float
    sat_mean: float
    val_mean: float
    hue_var: float
    sat_var: float
    val_var: float
    motion: float
    duration_sec: float


@dataclass(frozen=True)
class TrackEvent:
    start_sec: float
    dur_sec: float
    pitch: int
    amp: float
    attack: float
    decay: float
    rel: float


@dataclass(frozen=True)
class ReferenceRenderData:
    features: List[SegmentFeature]
    bpm: int
    reference_bpm: float
    render_bpm: float
    target_duration_sec: float
    raw_duration_sec: float
    scale_name: str
    mood_name: str
    global_activity: float
    melody_spb: int
    plan_count: int
    harmonic_plan_count: int
    harmony: List[TrackEvent]
    bass: List[TrackEvent]
    pad: List[TrackEvent]
    melody: List[TrackEvent]


@dataclass(frozen=True)
class MidiStemBundleData:
    render_bpm: float
    harmony_events: List[Tuple[int, Any]]
    bass_events: List[Tuple[int, Any]]
    pad_events: List[Tuple[int, Any]]
    melody_events: List[Tuple[int, Any]]


_V2A_CX_MODULE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert video to WAV using Nyquist for music generation and synthesis."
    )
    parser.add_argument("input", help="Input video path (or input directory in --batch mode).")
    parser.add_argument("output", help="Output WAV path (or output directory in --batch mode).")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process input-<n>.* videos in input directory to output-<n>.wav.",
    )

    parser.add_argument("--k", type=int, default=5, help="KMeans clusters when --method kmeans.")
    parser.add_argument("--method", choices=["kmeans", "hist"], default="kmeans", help="Feature reduction method.")
    parser.add_argument("--bins", type=int, default=36, help="Hue histogram bins when --method hist is selected.")

    parser.add_argument("--max-size", type=int, default=512, help="Resize longest frame side for speed.")
    parser.add_argument("--sample", type=int, default=60000, help="Max sampled pixels per segment.")
    parser.add_argument("--frames-per-segment", type=int, default=5, help="Number of sampled frames per segment.")
    parser.add_argument("--frame-step", type=int, default=1, help="Decode every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum sampled frames to decode.")
    parser.add_argument("--smooth", type=int, default=3, help="Segment smoothing window size.")

    parser.add_argument("--tempo", type=int, default=None, help="Manual BPM; omitted = Nyquist auto tempo.")
    parser.add_argument("--scale", choices=SCALES[1:], default=None, help="Manual scale; omitted = Nyquist auto scale.")
    parser.add_argument("--mood", choices=MOODS[1:], default=None, help="Mood preset.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and KMeans.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Nyquist mapping sensitivity.")
    parser.add_argument(
        "--mapping-bias",
        type=float,
        default=0.5,
        help="Blend between safety correction and direct visual mapping.",
    )

    # Compatibility flags from v2a-cx UI/CLI shape.
    parser.add_argument("--synth-output", action="store_true", help="Compatibility flag; Nyquist always renders WAV.")
    parser.add_argument("--direct-output", action="store_true", help="Compatibility flag; ignored in Nyquist mode.")
    parser.add_argument("--soundfont", type=str, default=None, help="Compatibility flag; ignored.")

    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate.")
    parser.add_argument("--nyquist-bin", type=str, default=None, help="Path to `ny`/`nyquist` binary.")
    parser.add_argument("--nyquist-timeout", type=int, default=900, help="Timeout seconds for Nyquist process.")
    return parser.parse_args()


def resize_pil(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = float(max_size) / float(max(w, h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def rgb_to_hsv_unit(img: Image.Image) -> np.ndarray:
    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.float32)
    arr[..., 0] /= 255.0
    arr[..., 1] /= 255.0
    arr[..., 2] /= 255.0
    return arr


def load_video_frames(path: str, max_size: int, frame_step: int, max_frames: int) -> Tuple[List[np.ndarray], float]:
    if imageio is None:
        raise RuntimeError("imageio is required for video decoding. Install with: pip install imageio imageio-ffmpeg")

    try:
        reader = imageio.get_reader(path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to open video: {path}") from exc

    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 0.0)
    if fps <= 0:
        fps = 30.0

    stride = max(1, int(frame_step))
    frames: List[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            if idx % stride != 0:
                continue
            pil = Image.fromarray(frame).convert("RGB")
            pil = resize_pil(pil, max_size=max(64, int(max_size)))
            hsv = rgb_to_hsv_unit(pil)
            frames.append(hsv)
            if len(frames) >= max(1, int(max_frames)):
                break
    finally:
        try:
            reader.close()
        except Exception:
            pass

    if not frames:
        raise RuntimeError("No frames decoded from input video.")
    return frames, fps


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def discover_batch_inputs(input_dir: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    for name in os.listdir(input_dir):
        m = INPUT_BATCH_RE.match(name)
        if m is None:
            continue
        items.append((int(m.group(1)), str(Path(input_dir) / name)))
    items.sort(key=lambda x: x[0])
    return items


def sample_pixels(hsv: np.ndarray, sample: int, rng: np.random.Generator) -> np.ndarray:
    pixels = hsv.reshape(-1, 3)
    n = int(pixels.shape[0])
    if n == 0:
        return pixels
    target = max(32, int(sample))
    if n <= target:
        return pixels
    idx = rng.choice(n, size=target, replace=False)
    return pixels[idx]


def circular_mean_unit(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if values.size == 0:
        return 0.0
    angles = values * (2.0 * math.pi)
    sinv = np.sin(angles)
    cosv = np.cos(angles)
    if weights is None:
        sx = float(np.mean(sinv))
        cx = float(np.mean(cosv))
    else:
        w = np.asarray(weights, dtype=np.float64)
        denom = float(np.sum(w))
        if denom <= 0:
            w = np.ones_like(values, dtype=np.float64)
            denom = float(w.size)
        sx = float(np.sum(sinv * w) / denom)
        cx = float(np.sum(cosv * w) / denom)
    angle = math.atan2(sx, cx)
    if angle < 0:
        angle += 2.0 * math.pi
    return angle / (2.0 * math.pi)


def circular_var_unit(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if values.size == 0:
        return 0.0
    angles = values * (2.0 * math.pi)
    sinv = np.sin(angles)
    cosv = np.cos(angles)
    if weights is None:
        sx = float(np.mean(sinv))
        cx = float(np.mean(cosv))
    else:
        w = np.asarray(weights, dtype=np.float64)
        denom = float(np.sum(w))
        if denom <= 0:
            w = np.ones_like(values, dtype=np.float64)
            denom = float(w.size)
        sx = float(np.sum(sinv * w) / denom)
        cx = float(np.sum(cosv * w) / denom)
    r = math.sqrt(sx * sx + cx * cx)
    return float(max(0.0, min(1.0, 1.0 - r)))


def circular_distance_unit(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    return min(d, 1.0 - d)


def normalize_series(values: Sequence[float]) -> List[float]:
    items = [float(v) for v in values]
    if not items:
        return []
    lo = min(items)
    hi = max(items)
    if hi - lo < 1e-9:
        return [0.0 for _ in items]
    return [(v - lo) / (hi - lo) for v in items]


def smooth_series(values: Sequence[float], window: int) -> List[float]:
    items = [float(v) for v in values]
    if len(items) < 2 or int(window) <= 1:
        return items
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    padded = [items[0]] * half + items + [items[-1]] * half
    out: List[float] = []
    for i in range(half, len(padded) - half):
        out.append(float(np.mean(padded[i - half : i + half + 1])))
    return out


def build_signal_s(frames: Sequence[np.ndarray], smooth_window: int) -> Tuple[List[float], List[float], List[float]]:
    if not frames:
        return [], [], []
    v_means = [float(np.mean(frame[..., 2])) for frame in frames]
    h_means = [float(circular_mean_unit(frame[..., 0].reshape(-1))) for frame in frames]

    dv = [0.0]
    dh = [0.0]
    for i in range(1, len(frames)):
        dv.append(abs(v_means[i] - v_means[i - 1]))
        dh.append(circular_distance_unit(h_means[i], h_means[i - 1]) / 0.5)

    dv_n = normalize_series(dv)
    dh_n = normalize_series(dh)
    s_raw = [0.6 * dv_n[i] + 0.4 * dh_n[i] for i in range(len(dv_n))]
    s_smooth = smooth_series(s_raw, smooth_window)
    return s_smooth, dv_n, dh_n


def segment_by_signal(signal: Sequence[float], min_len: int) -> List[Tuple[int, int]]:
    n = len(signal)
    if n == 0:
        return []
    min_len = max(2, int(min_len))
    if n <= min_len:
        return [(0, n)]

    threshold = float(np.quantile(signal, 0.4))
    candidates: List[int] = []
    for i in range(1, n - 1):
        if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1] and signal[i] <= threshold:
            candidates.append(i)

    boundaries = [0]
    last = 0
    for idx in candidates:
        if idx - last >= min_len:
            boundaries.append(idx)
            last = idx
    if n - boundaries[-1] < min_len and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(n)

    segments: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= 1:
            segments.append((start, end))
    if len(segments) <= 1:
        segments = []
        for start in range(0, n, min_len):
            end = min(n, start + min_len)
            segments.append((start, end))
    return segments


def frame_mean_hsv(frame: np.ndarray) -> Tuple[float, float, float]:
    pixels = frame.reshape(-1, 3)
    h = circular_mean_unit(pixels[:, 0])
    s = float(np.mean(pixels[:, 1])) if pixels.size else 0.0
    v = float(np.mean(pixels[:, 2])) if pixels.size else 0.0
    return h, s, v


def compute_histogram_cluster_statistics(pixels: np.ndarray, bins: int, k: int) -> Tuple[float, float, float, float, float, float]:
    if pixels.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]
    hist, bin_edges = np.histogram(hue, bins=max(1, int(bins)), range=(0.0, 1.0))
    peak_bins = np.argsort(hist)[::-1][: max(1, int(k))]

    hues: List[float] = []
    sats: List[float] = []
    vals: List[float] = []
    weights: List[float] = []
    total = float(pixels.shape[0])
    for b in peak_bins:
        low = float(bin_edges[b])
        high = float(bin_edges[b + 1])
        if int(b) == int(len(bin_edges) - 2):
            mask = (hue >= low) & (hue <= high)
        else:
            mask = (hue >= low) & (hue < high)
        if not np.any(mask):
            continue
        hues.append(float(np.mean(hue[mask])))
        sats.append(float(np.mean(sat[mask])))
        vals.append(float(np.mean(val[mask])))
        weights.append(float(np.sum(mask) / max(1.0, total)))

    if not weights:
        return (
            float(circular_mean_unit(hue)),
            float(np.mean(sat)),
            float(np.mean(val)),
            float(circular_var_unit(hue)),
            float(np.var(sat)),
            float(np.var(val)),
        )

    w = np.asarray(weights, dtype=np.float64)
    w = w / max(1e-9, float(np.sum(w)))
    hue_arr = np.asarray(hues, dtype=np.float64)
    sat_arr = np.asarray(sats, dtype=np.float64)
    val_arr = np.asarray(vals, dtype=np.float64)
    hue_mean = circular_mean_unit(hue_arr, weights=w)
    sat_mean = float(np.sum(sat_arr * w))
    val_mean = float(np.sum(val_arr * w))
    hue_var = circular_var_unit(hue_arr, weights=w)
    sat_var = float(np.sum(((sat_arr - sat_mean) ** 2) * w))
    val_var = float(np.sum(((val_arr - val_mean) ** 2) * w))
    return hue_mean, sat_mean, val_mean, hue_var, sat_var, val_var


def compute_segment_statistics(
    pixels: np.ndarray,
    *,
    method: str,
    k: int,
    bins: int,
    seed: int,
) -> Tuple[float, float, float, float, float, float]:
    if pixels.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    use_kmeans = method == "kmeans" and KMeans is not None and int(pixels.shape[0]) >= 3
    if use_kmeans:
        n_clusters = max(1, min(int(k), int(pixels.shape[0])))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed))
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_.astype(np.float64)
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        weights = counts / max(1.0, float(np.sum(counts)))
        hue_mean = circular_mean_unit(centers[:, 0], weights=weights)
        sat_mean = float(np.sum(centers[:, 1] * weights))
        val_mean = float(np.sum(centers[:, 2] * weights))
        hue_var = circular_var_unit(centers[:, 0], weights=weights)
        sat_var = float(np.sum(((centers[:, 1] - sat_mean) ** 2) * weights))
        val_var = float(np.sum(((centers[:, 2] - val_mean) ** 2) * weights))
        return hue_mean, sat_mean, val_mean, hue_var, sat_var, val_var

    if method == "hist":
        return compute_histogram_cluster_statistics(pixels, bins=bins, k=k)

    hue_mean = circular_mean_unit(pixels[:, 0])
    sat_mean = float(np.mean(pixels[:, 1]))
    val_mean = float(np.mean(pixels[:, 2]))
    hue_var = circular_var_unit(pixels[:, 0])
    sat_var = float(np.var(pixels[:, 1]))
    val_var = float(np.var(pixels[:, 2]))
    return hue_mean, sat_mean, val_mean, hue_var, sat_var, val_var


def compute_motion_score(frame_means: Sequence[Tuple[float, float, float]]) -> float:
    if len(frame_means) <= 1:
        return 0.0
    diffs: List[float] = []
    prev_h, prev_s, prev_v = frame_means[0]
    for h, s, v in frame_means[1:]:
        h_diff = circular_distance_unit(h, prev_h)
        s_diff = abs(s - prev_s)
        v_diff = abs(v - prev_v)
        diffs.append((h_diff + s_diff + v_diff) / 3.0)
        prev_h, prev_s, prev_v = h, s, v
    return float(max(0.0, min(1.0, np.mean(diffs) if diffs else 0.0)))


def build_segment_features(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    frame_step: int,
    segment_ranges: Sequence[Tuple[int, int]],
    method: str,
    k: int,
    bins: int,
    sample: int,
    seed: int,
) -> List[SegmentFeature]:
    features: List[SegmentFeature] = []
    sec_per_sampled_frame = max(1e-6, float(frame_step) / max(1e-6, float(fps)))

    for seg_idx, (start, end) in enumerate(segment_ranges):
        seg_frames = list(frames[int(start) : int(end)])
        if not seg_frames:
            continue
        merged = np.concatenate([f.reshape(-1, 3) for f in seg_frames], axis=0)
        sampled = sample_pixels(merged, sample=sample, rng=np.random.default_rng(int(seed + int(start))))
        hue_mean, sat_mean, val_mean, hue_var, sat_var, val_var = compute_segment_statistics(
            sampled, method=method, k=k, bins=bins, seed=seed + int(start)
        )
        frame_means = [frame_mean_hsv(f) for f in seg_frames]
        motion = compute_motion_score(frame_means)
        duration_sec = float(len(seg_frames)) * sec_per_sampled_frame
        features.append(
            SegmentFeature(
                hue_mean=float(np.clip(hue_mean, 0.0, 1.0)),
                sat_mean=float(np.clip(sat_mean, 0.0, 1.0)),
                val_mean=float(np.clip(val_mean, 0.0, 1.0)),
                hue_var=float(np.clip(hue_var, 0.0, 1.0)),
                sat_var=float(max(0.0, sat_var)),
                val_var=float(max(0.0, val_var)),
                motion=float(np.clip(motion, 0.0, 1.0)),
                duration_sec=float(max(0.05, duration_sec)),
            )
        )
    return features


def smooth_segment_features(features: Sequence[SegmentFeature], window: int) -> List[SegmentFeature]:
    items = list(features)
    if not items:
        return items
    w = max(1, int(window))
    if w <= 1:
        return items
    radius = w // 2
    out: List[SegmentFeature] = []

    for i in range(len(items)):
        left = max(0, i - radius)
        right = min(len(items), i + radius + 1)
        block = items[left:right]
        hues = np.array([x.hue_mean for x in block], dtype=np.float64)
        out.append(
            SegmentFeature(
                hue_mean=circular_mean_unit(hues),
                sat_mean=float(np.mean([x.sat_mean for x in block])),
                val_mean=float(np.mean([x.val_mean for x in block])),
                hue_var=float(np.mean([x.hue_var for x in block])),
                sat_var=float(np.mean([x.sat_var for x in block])),
                val_var=float(np.mean([x.val_var for x in block])),
                motion=float(np.mean([x.motion for x in block])),
                duration_sec=items[i].duration_sec,
            )
        )
    return out


def hue_to_wavelength_nm_unit(hue_unit: float) -> float:
    hue = float(hue_unit % 1.0) * 360.0
    return 750.0 - (hue / 360.0) * 370.0


def optical_frequency_param_unit(hue_unit: float) -> float:
    wavelength_nm = hue_to_wavelength_nm_unit(hue_unit)
    wavelength_m = wavelength_nm * 1e-9
    f_light = C_M_PER_S / wavelength_m
    return math.log(f_light / 1e4)


def optical_frequency_param_from_wavelength_nm(wavelength_nm: float) -> float:
    wavelength_m = float(wavelength_nm) * 1e-9
    f_light = C_M_PER_S / wavelength_m
    return math.log(f_light / 1e4)


F_PARAM_MIN = optical_frequency_param_from_wavelength_nm(750.0)
F_PARAM_MAX = optical_frequency_param_from_wavelength_nm(380.0)
F_PARAM_RANGE = max(1e-6, F_PARAM_MAX - F_PARAM_MIN)


def compute_cluster_summaries(
    pixels: np.ndarray,
    *,
    method: str,
    k: int,
    bins: int,
    seed: int,
) -> List[Tuple[float, float, float, float, float]]:
    if pixels.size == 0:
        return []

    use_kmeans = method == "kmeans" and KMeans is not None and int(pixels.shape[0]) >= 3
    if use_kmeans:
        n_clusters = max(1, min(int(k), int(pixels.shape[0])))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed))
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_.astype(np.float64)
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        weights = counts / max(1.0, float(np.sum(counts)))
        out: List[Tuple[float, float, float, float, float]] = []
        for i in range(n_clusters):
            out.append(
                (
                    float(centers[i, 0]),
                    float(centers[i, 1]),
                    float(centers[i, 2]),
                    float(weights[i]),
                    float(optical_frequency_param_unit(float(centers[i, 0]))),
                )
            )
        out.sort(key=lambda item: item[3], reverse=True)
        return out

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]
    hist, bin_edges = np.histogram(hue, bins=max(1, int(bins)), range=(0.0, 1.0))
    peak_bins = np.argsort(hist)[::-1][: max(1, int(k))]

    out = []
    total = float(pixels.shape[0])
    for b in peak_bins:
        low = float(bin_edges[b])
        high = float(bin_edges[b + 1])
        if int(b) == int(len(bin_edges) - 2):
            mask = (hue >= low) & (hue <= high)
        else:
            mask = (hue >= low) & (hue < high)
        if not np.any(mask):
            continue
        h = float(np.mean(hue[mask]))
        s = float(np.mean(sat[mask]))
        v = float(np.mean(val[mask]))
        weight = float(np.sum(mask) / max(1.0, total))
        out.append((h, s, v, weight, float(optical_frequency_param_unit(h))))
    out.sort(key=lambda item: item[3], reverse=True)
    return out


def luminance_entropy(pixels: np.ndarray, bins: int = 32) -> float:
    if pixels.size == 0:
        return 0.0
    v = pixels[:, 2]
    hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    max_entropy = math.log2(bins) if bins > 1 else 1.0
    return float(entropy / max_entropy)


def optical_span_norm(clusters: Sequence[Tuple[float, float, float, float, float]]) -> float:
    if not clusters:
        return 0.0
    vals = [float(item[4]) for item in clusters]
    span = max(vals) - min(vals)
    return float(max(0.0, min(1.0, span / F_PARAM_RANGE)))


def hue_warmth_score_unit(hue_unit: float) -> float:
    hue = (float(hue_unit) % 1.0) * 360.0
    return float(max(0.0, min(1.0, 0.5 + 0.5 * math.cos(math.radians(hue - 45.0)))))


def affect_from_segment_feature(feature: SegmentFeature) -> Tuple[float, float]:
    sat = float(np.clip(feature.sat_mean, 0.0, 1.0))
    val = float(np.clip(feature.val_mean, 0.0, 1.0))
    warm = hue_warmth_score_unit(feature.hue_mean)
    motion = float(np.clip(0.6 * feature.hue_var + 1.8 * feature.sat_var + 1.8 * feature.val_var, 0.0, 1.0))
    valence = float(np.clip(0.52 * val + 0.30 * sat + 0.18 * warm, 0.0, 1.0))
    arousal = float(np.clip(0.48 * sat + 0.28 * (1.0 - val) + 0.24 * motion, 0.0, 1.0))
    return valence, arousal


def global_affect(features: Sequence[SegmentFeature]) -> Tuple[float, float]:
    if not features:
        return 0.5, 0.5
    total_w = 0.0
    val_sum = 0.0
    aro_sum = 0.0
    for feat in features:
        w = float(max(1e-6, feat.duration_sec))
        v, a = affect_from_segment_feature(feat)
        total_w += w
        val_sum += v * w
        aro_sum += a * w
    if total_w <= 1e-9:
        return 0.5, 0.5
    return float(np.clip(val_sum / total_w, 0.0, 1.0)), float(np.clip(aro_sum / total_w, 0.0, 1.0))


def select_bpm(entropy_norm: float, r_opt_norm: float) -> int:
    score = (float(entropy_norm) + float(r_opt_norm)) / 2.0
    if entropy_norm < 0.35 and r_opt_norm < 0.35:
        bpm = 60 + 20 * score
    elif entropy_norm < 0.65:
        bpm = 80 + 30 * score
    else:
        bpm = 110 + 30 * score
    return int(max(40, min(200, round(bpm))))


def select_bpm_affective(arousal: float) -> int:
    bpm = 55.0 + 95.0 * float(np.clip(arousal, 0.0, 1.0))
    return int(max(45, min(185, round(bpm))))


def optical_trend(features: Sequence[SegmentFeature]) -> float:
    vals = [optical_frequency_param_unit(feat.hue_mean) for feat in features]
    if len(vals) < 2:
        return 0.5
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-6:
        return 0.5
    mean = float(np.mean(vals))
    return float((mean - lo) / (hi - lo))


def choose_scale(features: Sequence[SegmentFeature], override: Optional[str] = None) -> str:
    if override:
        return str(override)
    if not features:
        return "major"

    avg_sat = float(np.mean([f.sat_mean for f in features]))
    avg_val = float(np.mean([f.val_mean for f in features]))
    if avg_sat < 0.3:
        return "pentatonic"

    avg_hue = circular_mean_unit(np.asarray([f.hue_mean for f in features], dtype=np.float64)) * 360.0
    if avg_hue < 60.0 or avg_hue >= 300.0:
        base = "major"
    elif avg_hue < 180.0:
        base = "dorian"
    else:
        base = "minor"

    if avg_val >= 0.60 and base in ("minor", "dorian"):
        base = "major"
    if avg_val <= 0.33 and base == "major":
        base = "dorian"

    trend = optical_trend(features)
    if base == "minor" and trend > 0.66:
        return "mixolydian"
    if base == "major" and trend < 0.33:
        return "dorian"
    return base


def resolve_reference_context(
    *,
    frames: Sequence[np.ndarray],
    features: Sequence[SegmentFeature],
    sample: int,
    method: str,
    k: int,
    bins: int,
    seed: int,
) -> Tuple[int, str]:
    if not features:
        return 120, "major"
    global_pixels = sample_pixels(
        np.concatenate([frame.reshape(-1, 3) for frame in frames], axis=0),
        sample=max(256, int(sample)),
        rng=np.random.default_rng(int(seed)),
    )
    global_clusters = compute_cluster_summaries(global_pixels, method=method, k=k, bins=bins, seed=seed)
    entropy_norm = luminance_entropy(global_pixels)
    r_opt_norm = optical_span_norm(global_clusters)
    auto_bpm = select_bpm(entropy_norm, r_opt_norm)
    _, global_arousal = global_affect(features)
    affective_bpm = select_bpm_affective(global_arousal)
    bpm = int(max(40, min(200, round(0.35 * auto_bpm + 0.65 * affective_bpm))))
    scale_name = choose_scale(features, None)
    return bpm, scale_name


def load_v2a_cx_module():
    global _V2A_CX_MODULE
    if _V2A_CX_MODULE is not None:
        return _V2A_CX_MODULE
    cx_path = Path(__file__).resolve().parent / "composition_engine.py"
    if not cx_path.exists():
        raise RuntimeError(f"Reference pipeline not found: {cx_path}")
    spec = importlib.util.spec_from_file_location("v2a_cx_reference", str(cx_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reference pipeline: {cx_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(cx_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    _V2A_CX_MODULE = module
    return module


def velocity_to_amp(track_name: str, velocity: int) -> float:
    base = {
        "harmony": 0.22,
        "bass": 0.24,
        "pad": 0.12,
        "melody": 0.24,
    }.get(str(track_name), 0.20)
    vel = float(np.clip(float(velocity) / 127.0, 0.0, 1.0))
    return float(base * (0.30 + 0.70 * vel))


def track_events_from_intervals(reference_module, events, *, seconds_per_tick: float, track_name: str) -> List[TrackEvent]:
    sec_per_tick = float(max(1e-9, seconds_per_tick))
    env = {
        "harmony": (0.03, 0.08, 0.12),
        "bass": (0.02, 0.08, 0.12),
        "pad": (0.04, 0.12, 0.14),
        "melody": (0.02, 0.10, 0.14),
    }.get(str(track_name), (0.03, 0.08, 0.12))
    out: List[TrackEvent] = []
    for start_tick, end_tick, note, velocity in reference_module.collect_note_intervals(events):
        start_sec = float(start_tick) * sec_per_tick
        dur_sec = max(sec_per_tick, float(end_tick - start_tick) * sec_per_tick)
        out.append(
            TrackEvent(
                start_sec=float(max(0.0, start_sec)),
                dur_sec=float(max(0.01, dur_sec)),
                pitch=int(note),
                amp=float(max(0.01, velocity_to_amp(track_name, int(velocity)))),
                attack=float(env[0]),
                decay=float(env[1]),
                rel=float(env[2]),
            )
        )
    return out


def max_event_end_tick(reference_module, *event_groups) -> int:
    max_end = 0
    for events in event_groups:
        for _, end_tick, _, _ in reference_module.collect_note_intervals(events):
            if int(end_tick) > max_end:
                max_end = int(end_tick)
    return int(max_end)


def build_reference_render_data(
    *,
    args: argparse.Namespace,
    frames: Sequence[np.ndarray],
    fps: float,
    signal_s: Sequence[float],
    segment_ranges: Sequence[Tuple[int, int]],
    target_duration_sec: float,
) -> Tuple[ReferenceRenderData, MidiStemBundleData]:
    cx = load_v2a_cx_module()
    reference_frames = []
    for frame in frames:
        ref_frame = np.array(frame, dtype=np.float32, copy=True)
        ref_frame[..., 0] *= 360.0
        reference_frames.append(ref_frame)
    signal_ref, _, _ = cx.build_signal_s(reference_frames, max(1, int(args.smooth)))
    mood_settings = cx.MOODS.get(args.mood, {}) if args.mood else {}
    global_pixels = cx.sample_pixels(
        np.concatenate([frame.reshape(-1, 3) for frame in reference_frames], axis=0),
        max(256, int(args.sample)),
        int(args.seed),
    )
    if str(args.method) == "kmeans":
        global_clusters = cx.compute_clusters_kmeans(global_pixels, int(args.k), int(args.seed))
    else:
        global_clusters = cx.compute_clusters_histogram(global_pixels, int(args.bins), int(args.k))

    entropy_norm = cx.luminance_entropy(global_pixels)
    r_opt_norm = cx.optical_span_norm(global_clusters)
    auto_bpm = cx.select_bpm(entropy_norm, r_opt_norm)
    mu = float(np.mean(signal_ref)) if signal_ref else 0.0
    sigma = float(np.var(signal_ref)) if signal_ref else 0.0
    signal_hash = int(np.sum(np.round(np.asarray(signal_ref, dtype=np.float64) * 1000.0))) & 0xFFFFFFFF
    seed = int((int(args.seed) * 1315423911 + signal_hash + len(signal_ref) * 2654435761) & 0xFFFFFFFF)
    accent = cx.accent_strength(signal_ref)

    ref_features = []
    filtered_ranges: List[Tuple[int, int]] = []
    for start, end in segment_ranges:
        block = list(reference_frames[int(start) : int(end)])
        if not block:
            continue
        merged = np.concatenate([frame.reshape(-1, 3) for frame in block], axis=0)
        pixels = cx.sample_pixels(merged, int(args.sample), int(args.seed) + int(start))
        if str(args.method) == "kmeans":
            clusters = cx.compute_clusters_kmeans(pixels, int(args.k), int(args.seed) + int(start))
        else:
            clusters = cx.compute_clusters_histogram(pixels, int(args.bins), int(args.k))
        ref_features.append(cx.build_segment_features(clusters))
        filtered_ranges.append((int(start), int(end)))
    if not ref_features:
        raise RuntimeError("Reference pipeline produced no segment features.")

    ref_features = cx.smooth_segments(ref_features, max(1, int(args.smooth)))
    segment_infos = [
        cx.SegmentInfo(features=feat, start=int(start), end=int(end))
        for (start, end), feat in zip(filtered_ranges, ref_features)
    ]
    scale_override = args.scale if args.scale else mood_settings.get("scale")
    scale_name = cx.choose_scale([seg.features for seg in segment_infos], scale_override)
    global_valence, global_arousal = cx.global_affect(segment_infos)
    affective_bpm = cx.select_bpm_affective(global_arousal)
    auto_bpm = cx.clamp_int(int(round(0.35 * auto_bpm + 0.65 * affective_bpm)), 40, 200)

    tempo_bpm = int(args.tempo) if args.tempo is not None else int(auto_bpm)
    if args.tempo is None and mood_settings.get("tempo") is not None:
        tempo_bpm = int(round(float(tempo_bpm) * (float(mood_settings["tempo"]) / 120.0)))
    tempo_bpm = int(cx.clamp_int(tempo_bpm, 40, 200))

    scale = cx.SCALES[scale_name]
    ticks_per_beat = 480
    ticks_per_second = ticks_per_beat * (tempo_bpm / 60.0)
    ticks_per_frame = ticks_per_second / max(1e-6, float(fps))
    total_ticks = int(max(1, round(len(frames) * ticks_per_frame)))
    global_activity = float(cx._clamp01(mu + sigma * 2.5))
    _, global_sync, _ = cx.rhythm_targets_from_affect(global_valence, global_arousal, global_activity)
    melody_spb = int(cx.choose_global_rhythm_steps_per_beat(global_activity, global_sync))
    harmony_spb = (2, 6)
    bass_pad_spb = 1
    phrase_beats = 16.0 if global_activity < 0.45 else 8.0
    tensions = cx.normalize([seg.features.f_param_mean for seg in segment_infos])
    tension_factor = float(mood_settings.get("tension_factor", 1.0))

    segment_plans = cx.build_segment_module_plans(
        segments=segment_infos,
        scale=scale,
        tensions=tensions,
        tension_factor=tension_factor,
        sensitivity=float(max(0.05, args.sensitivity)),
        mood_settings=mood_settings,
        ticks_per_frame=ticks_per_frame,
        bars_per_segment=1,
        ticks_per_beat=ticks_per_beat,
        rhythm_steps_per_beat=harmony_spb,
    )
    harmonic_plans = cx.adapt_harmonic_progression_rate(
        segment_plans,
        ticks_per_beat=ticks_per_beat,
        rhythm_steps_per_beat=harmony_spb,
    )

    harmony_events = []
    bass_events = []
    pad_events = []
    melody_events = []
    ch_harmony = 0
    ch_bass = 1
    ch_pad = 2
    ch_melody = 3

    cx.emit_harmony_module(harmony_events, harmonic_plans, channel=ch_harmony)
    cx.emit_bass_module(bass_events, harmonic_plans, channel=ch_bass)
    cx.emit_pad_module(pad_events, harmonic_plans, channel=ch_pad)
    cx.emit_melody_module(
        melody_events,
        segments=segment_infos,
        accent=accent,
        harmony_events=harmony_events,
        bass_events=bass_events,
        pad_events=pad_events,
        total_frames=len(frames),
        ticks_per_frame=ticks_per_frame,
        total_ticks=total_ticks,
        ticks_per_beat=ticks_per_beat,
        phrase_beats=phrase_beats,
        seed=seed,
        mu=mu,
        sigma=sigma,
        sensitivity=float(max(0.05, args.sensitivity)),
        mood_settings=mood_settings,
        scale=scale,
        channel=ch_melody,
        rhythm_steps_per_beat=melody_spb,
    )

    harmony_events = cx.quantize_track_note_events(
        harmony_events,
        ticks_per_beat,
        ch_harmony,
        cx.HARMONY_RANGE[0],
        cx.HARMONY_RANGE[1],
        rhythm_steps_per_beat=harmony_spb,
    )
    bass_events = cx.quantize_track_note_events(
        bass_events,
        ticks_per_beat,
        ch_bass,
        cx.BASS_RANGE[0],
        cx.BASS_RANGE[1],
        rhythm_steps_per_beat=bass_pad_spb,
    )
    pad_events = cx.quantize_track_note_events(
        pad_events,
        ticks_per_beat,
        ch_pad,
        cx.PAD_RANGE[0],
        cx.PAD_RANGE[1],
        rhythm_steps_per_beat=bass_pad_spb,
    )
    melody_events = cx.quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        cx.MELODY_RANGE[0],
        cx.MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_spb,
    )

    def scale_pcs_at_tick(tick: int) -> set:
        _ = tick
        return {int(deg) % 12 for deg in scale}

    melody_events = cx.resolve_melody_clashes_after_quantization(
        melody_events,
        harmony_events,
        bass_events,
        pad_events,
        ticks_per_beat=ticks_per_beat,
        melody_channel=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick,
        rhythm_steps_per_beat=melody_spb,
    )
    harmony_events = cx.quantize_track_note_events(
        harmony_events,
        ticks_per_beat,
        ch_harmony,
        cx.HARMONY_RANGE[0],
        cx.HARMONY_RANGE[1],
        rhythm_steps_per_beat=harmony_spb,
    )
    bass_events = cx.quantize_track_note_events(
        bass_events,
        ticks_per_beat,
        ch_bass,
        cx.BASS_RANGE[0],
        cx.BASS_RANGE[1],
        rhythm_steps_per_beat=bass_pad_spb,
    )
    pad_events = cx.quantize_track_note_events(
        pad_events,
        ticks_per_beat,
        ch_pad,
        cx.PAD_RANGE[0],
        cx.PAD_RANGE[1],
        rhythm_steps_per_beat=bass_pad_spb,
    )
    melody_events = cx.quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        cx.MELODY_RANGE[0],
        cx.MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_spb,
    )
    harmony_events, bass_events, pad_events, melody_events, _, _ = cx.maybe_apply_safe_post_fallback(
        harmony_events=harmony_events,
        bass_events=bass_events,
        pad_events=pad_events,
        melody_events=melody_events,
        ticks_per_beat=ticks_per_beat,
        ch_harmony=ch_harmony,
        ch_bass=ch_bass,
        ch_pad=ch_pad,
        ch_melody=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick,
        mapping_bias=float(getattr(args, "mapping_bias", 0.5)),
    )
    melody_events = cx.resolve_melody_clashes_after_quantization(
        melody_events,
        harmony_events,
        bass_events,
        pad_events,
        ticks_per_beat=ticks_per_beat,
        melody_channel=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick,
        rhythm_steps_per_beat=melody_spb,
    )
    melody_events = cx.quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        cx.MELODY_RANGE[0],
        cx.MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_spb,
    )

    raw_end_tick = max(
        max_event_end_tick(cx, harmony_events, bass_events, pad_events, melody_events),
        int(total_ticks),
        1,
    )
    raw_duration_sec = float(raw_end_tick) * ((60.0 / max(1, int(tempo_bpm))) / float(max(1, int(ticks_per_beat))))
    target_duration = float(max(1e-6, target_duration_sec))
    seconds_per_tick = float(target_duration / float(max(1, raw_end_tick)))
    render_bpm = float(60.0 / max(1e-9, seconds_per_tick * float(max(1, int(ticks_per_beat)))))

    summary_features: List[SegmentFeature] = []
    sec_per_sampled_frame = max(1e-6, float(max(1, int(args.frame_step))) / max(1e-6, float(fps)))
    for (start, end), feat in zip(filtered_ranges, ref_features):
        block = list(frames[int(start) : int(end)])
        frame_means = [frame_mean_hsv(frame) for frame in block]
        summary_features.append(
            SegmentFeature(
                hue_mean=float((float(feat.hue_mean) % 360.0) / 360.0),
                sat_mean=float(np.clip(feat.sat_mean, 0.0, 1.0)),
                val_mean=float(np.clip(feat.val_mean, 0.0, 1.0)),
                hue_var=float(np.clip(feat.hue_var, 0.0, 1.0)),
                sat_var=float(max(0.0, feat.sat_var)),
                val_var=float(max(0.0, feat.val_var)),
                motion=float(np.clip(compute_motion_score(frame_means), 0.0, 1.0)),
                duration_sec=float(max(0.05, (int(end) - int(start)) * sec_per_sampled_frame)),
            )
        )

    return (
        ReferenceRenderData(
            features=summary_features,
            bpm=int(round(render_bpm)),
            reference_bpm=float(tempo_bpm),
            render_bpm=float(render_bpm),
            target_duration_sec=float(target_duration),
            raw_duration_sec=float(raw_duration_sec),
            scale_name=str(scale_name),
            mood_name=symbol_for_mood(args.mood),
            global_activity=float(global_activity),
            melody_spb=int(melody_spb),
            plan_count=len(segment_plans),
            harmonic_plan_count=len(harmonic_plans),
            harmony=track_events_from_intervals(cx, harmony_events, seconds_per_tick=seconds_per_tick, track_name="harmony"),
            bass=track_events_from_intervals(cx, bass_events, seconds_per_tick=seconds_per_tick, track_name="bass"),
            pad=track_events_from_intervals(cx, pad_events, seconds_per_tick=seconds_per_tick, track_name="pad"),
            melody=track_events_from_intervals(cx, melody_events, seconds_per_tick=seconds_per_tick, track_name="melody"),
        ),
        MidiStemBundleData(
            render_bpm=float(render_bpm),
            harmony_events=list(harmony_events),
            bass_events=list(bass_events),
            pad_events=list(pad_events),
            melody_events=list(melody_events),
        ),
    )


def lisp_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_lisp_float(x: float) -> str:
    return f"{float(x):.8f}"


def write_nyquist_data(path: Path, features: Sequence[SegmentFeature]) -> None:
    lines = ["(setf *v2a-segments* '("]
    for feat in features:
        row = [
            to_lisp_float(feat.hue_mean),
            to_lisp_float(feat.sat_mean),
            to_lisp_float(feat.val_mean),
            to_lisp_float(feat.hue_var),
            to_lisp_float(feat.sat_var),
            to_lisp_float(feat.val_var),
            to_lisp_float(feat.motion),
            to_lisp_float(feat.duration_sec),
        ]
        lines.append("  (" + " ".join(row) + ")")
    lines.append("))")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_nyquist_reference_data(path: Path, render_data: ReferenceRenderData) -> None:
    def emit_track(lines: List[str], var_name: str, events: Sequence[TrackEvent]) -> None:
        lines.append(f"(setf {var_name} '(")
        for ev in events:
            row = [
                to_lisp_float(ev.start_sec),
                to_lisp_float(ev.dur_sec),
                str(int(ev.pitch)),
                to_lisp_float(ev.amp),
                to_lisp_float(ev.attack),
                to_lisp_float(ev.decay),
                to_lisp_float(ev.rel),
            ]
            lines.append("  (" + " ".join(row) + ")")
        lines.append("))")

    lines = [
        "(setf *v2a-meta* '(",
        f"  (segments {int(len(render_data.features))})",
        f"  (bpm {int(render_data.bpm)})",
        f"  (reference-bpm {to_lisp_float(render_data.reference_bpm)})",
        f"  (render-bpm {to_lisp_float(render_data.render_bpm)})",
        f"  (target-duration {to_lisp_float(render_data.target_duration_sec)})",
        f"  (raw-duration {to_lisp_float(render_data.raw_duration_sec)})",
        f"  (scale {render_data.scale_name})",
        f"  (mood {render_data.mood_name})",
        f"  (activity {to_lisp_float(render_data.global_activity)})",
        f"  (melody-spb {int(render_data.melody_spb)})",
        f"  (plans {int(render_data.plan_count)})",
        f"  (harmonic-plans {int(render_data.harmonic_plan_count)})",
        "))",
    ]
    emit_track(lines, "*v2a-harmony-events*", render_data.harmony)
    emit_track(lines, "*v2a-bass-events*", render_data.bass)
    emit_track(lines, "*v2a-pad-events*", render_data.pad)
    emit_track(lines, "*v2a-melody-events*", render_data.melody)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_feature_spread(features: Sequence[SegmentFeature]) -> str:
    if not features:
        return "Feature spread: unavailable (no segments)."
    arr = np.array(
        [[f.hue_mean, f.sat_mean, f.val_mean, f.hue_var, f.sat_var, f.val_var, f.motion] for f in features],
        dtype=np.float64,
    )
    hue_std = float(np.std(arr[:, 0]))
    sat_std = float(np.std(arr[:, 1]))
    val_std = float(np.std(arr[:, 2]))
    motion_mean = float(np.mean(arr[:, 6]))
    motion_std = float(np.std(arr[:, 6]))
    return (
        "Feature spread: "
        f"hue_std={hue_std:.4f} sat_std={sat_std:.4f} val_std={val_std:.4f} "
        f"motion_mean={motion_mean:.4f} motion_std={motion_std:.4f}"
    )


def resolve_nyquist_bin(cli_value: Optional[str]) -> str:
    candidates: Iterable[Optional[str]] = (
        cli_value,
        os.environ.get("NYQUIST_BIN"),
        shutil.which("ny"),
        shutil.which("nyquist"),
        "/Applications/NyquistIDE.app/Contents/Java/ny",
    )
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.exists():
            return str(p)
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
    raise RuntimeError(
        "Nyquist binary not found. Install Nyquist and ensure `ny` or `nyquist` is in PATH, "
        "or pass --nyquist-bin /path/to/ny"
    )


def ensure_wav_path(path_text: str) -> str:
    p = Path(path_text)
    low = p.name.lower()
    if low.endswith(".wav"):
        out = p
    elif low.endswith(".mid") or low.endswith(".midi"):
        out = p.with_suffix(".wav")
    else:
        out = Path(str(p) + ".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def build_midi_stem_zip_path(audio_output_path: str) -> str:
    wav_path = Path(ensure_wav_path(audio_output_path)).expanduser().resolve()
    return str(wav_path.with_name(f"{wav_path.stem}_midi_stems.zip"))


def export_midi_stem_bundle(bundle: MidiStemBundleData, *, output_zip_path: str, tempo_bpm: float) -> str:
    cx = load_v2a_cx_module()
    zip_path = Path(output_zip_path).expanduser().resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [
        ("harmonics.mid", "Harmonics", "Harmonics", 0, bundle.harmony_events),
        ("pad.mid", "Pad", "Pad", 48, bundle.pad_events),
        ("bass.mid", "Bass", "Bass", 32, bundle.bass_events),
        ("melody.mid", "Melody", "Melody", 73, bundle.melody_events),
    ]

    # Export the quantized pre-synthesis note grid directly so the MIDI stems
    # keep exact note timing, velocities, and tempo metadata for DAW import.
    with tempfile.TemporaryDirectory(prefix=f"{zip_path.stem}_", dir=str(zip_path.parent)) as td:
        td_path = Path(td)
        for filename, track_name, instrument_name, program, events in tracks:
            cx.write_single_track_midi(
                str(td_path / filename),
                list(events),
                tempo_bpm=float(max(1.0, tempo_bpm)),
                track_name=track_name,
                instrument_name=instrument_name,
                program=int(program),
                auto_instrument=False,
            )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename, _, _, _, _ in tracks:
                archive.write(td_path / filename, arcname=filename)

    return str(zip_path)


def symbol_for_scale(value: Optional[str]) -> str:
    scale = (value or "auto").strip().lower()
    if scale not in SCALES:
        scale = "auto"
    return scale


def symbol_for_mood(value: Optional[str]) -> str:
    mood = (value or "none").strip().lower()
    if mood not in MOODS:
        mood = "none"
    return mood


def build_driver_script(
    *,
    driver_path: Path,
    engine_lsp_path: Path,
    data_path: Path,
    output_path: str,
    sample_rate: int,
) -> None:
    content = f"""
(load "{lisp_escape(str(engine_lsp_path))}")
(load "{lisp_escape(str(data_path))}")
(v2a-render-from-track-data
  "{lisp_escape(output_path)}"
  {int(sample_rate)})
(exit)
""".strip()
    driver_path.write_text(content + "\n", encoding="utf-8")


def run_nyquist_script(
    *,
    nyquist_bin: str,
    driver_path: Path,
    cwd: Path,
    timeout_sec: int,
    expected_output_path: Optional[str] = None,
) -> str:
    # Prefer explicit load form first; positional script path behaves inconsistently
    # across Nyquist builds/environments.
    attempts = [
        [nyquist_bin, "-l", str(driver_path)],
        [nyquist_bin, str(driver_path)],
    ]
    errors: List[str] = []
    for cmd in attempts:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                # Force EOF on stdin so Nyquist cannot hang on REPL prompt after an error.
                input="",
                timeout=max(30, int(timeout_sec)),
            )
        except subprocess.TimeoutExpired as exc:
            errors.append(f"Timeout: {' '.join(cmd)}\n{exc}")
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        low = out.lower()
        has_nyq_error = ("error:" in low) or ("stack overflow" in low)
        wrote_marker = "nyq | wrote=" in low
        output_exists = bool(expected_output_path) and Path(str(expected_output_path)).exists()

        if proc.returncode == 0 and (not has_nyq_error) and (wrote_marker or output_exists):
            return out.strip()
        errors.append(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"Nyquist error detected={has_nyq_error} wrote_marker={wrote_marker} output_exists={output_exists}\n"
            f"{out}"
        )
    raise RuntimeError("\n\n".join(errors))


def run_single_pipeline(args: argparse.Namespace, *, input_path: str, output_path: str, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"{tag}Decoding video: {input_path}", flush=True)
    target_duration_sec = probe_video_duration_seconds(input_path)
    frames, fps = load_video_frames(
        input_path,
        max_size=max(64, int(args.max_size)),
        frame_step=max(1, int(args.frame_step)),
        max_frames=max(1, int(args.max_frames)),
    )
    print(f"{tag}Frames decoded: {len(frames)} | fps={fps:.3f}", flush=True)

    print(f"{tag}Extracting HSV segment features...", flush=True)
    signal_s, _, _ = build_signal_s(frames, max(1, int(args.smooth)))
    segment_ranges = segment_by_signal(signal_s, max(1, int(args.frames_per_segment)))
    if not segment_ranges:
        segment_ranges = [(0, len(frames))]
    reference_data, midi_bundle_data = build_reference_render_data(
        args=args,
        frames=frames,
        fps=fps,
        signal_s=signal_s,
        segment_ranges=segment_ranges,
        target_duration_sec=target_duration_sec,
    )
    print(f"{tag}Segments prepared: {len(reference_data.features)}", flush=True)
    print(f"{tag}{summarize_feature_spread(reference_data.features)}", flush=True)
    print(
        f"{tag}Reference context: bpm={int(round(reference_data.reference_bpm))} "
        f"render_bpm={reference_data.render_bpm:.3f} scale={reference_data.scale_name} "
        f"plans={reference_data.plan_count}/{reference_data.harmonic_plan_count} "
        f"target={reference_data.target_duration_sec:.3f}s raw={reference_data.raw_duration_sec:.3f}s",
        flush=True,
    )

    final_output = ensure_wav_path(output_path)
    root = Path(__file__).resolve().parent
    engine_lsp = root / "v2a_nyq_engine.lsp"
    if not engine_lsp.exists():
        raise RuntimeError(f"Nyquist engine not found: {engine_lsp}")

    nyquist_bin = resolve_nyquist_bin(args.nyquist_bin)
    print(f"{tag}Using Nyquist binary: {nyquist_bin}", flush=True)

    with tempfile.TemporaryDirectory(prefix="v2a_nyq_") as td:
        td_path = Path(td)
        data_file = td_path / "track_data.lsp"
        driver_file = td_path / "driver_job.lsp"
        write_nyquist_reference_data(data_file, reference_data)
        build_driver_script(
            driver_path=driver_file,
            engine_lsp_path=engine_lsp,
            data_path=data_file,
            output_path=final_output,
            sample_rate=int(max(8000, args.sample_rate)),
        )

        print(f"{tag}Running Nyquist composition + synthesis...", flush=True)
        nyq_logs = run_nyquist_script(
            nyquist_bin=nyquist_bin,
            driver_path=driver_file,
            cwd=root,
            timeout_sec=int(args.nyquist_timeout),
            expected_output_path=final_output,
        )
        if nyq_logs.strip():
            print(nyq_logs.strip(), flush=True)

    if not Path(final_output).exists():
        raise RuntimeError(f"Nyquist finished but output was not found: {final_output}")
    sync_info = sync_wav_to_video_duration(final_output, input_path)
    video_duration = float(sync_info["video_duration"])
    audio_before = float(sync_info["audio_duration_before"])
    audio_after = float(sync_info["audio_duration_after"])
    delta_after = float(sync_info["delta_after"])
    if bool(sync_info["modified"]):
        print(
            f"{tag}Duration sync: video={video_duration:.3f}s audio_before={audio_before:.3f}s "
            f"audio_after={audio_after:.3f}s tempo_factor={float(sync_info['tempo_factor']):.6f}",
            flush=True,
        )
    else:
        print(
            f"{tag}Duration sync skipped: video={video_duration:.3f}s audio={audio_after:.3f}s "
            f"within tolerance",
            flush=True,
        )
    final_audio_duration = probe_wav_duration_seconds(final_output)
    final_video_duration = probe_video_duration_seconds(input_path)
    final_delta = abs(final_audio_duration - final_video_duration)
    print(
        f"{tag}Duration check: video={final_video_duration:.3f}s audio={final_audio_duration:.3f}s "
        f"diff={final_delta:.3f}s",
        flush=True,
    )
    if final_delta > 0.05 or delta_after > 0.05:
        raise RuntimeError(
            f"Synced WAV duration still mismatches video: video={final_video_duration:.3f}s "
            f"audio={final_audio_duration:.3f}s diff={final_delta:.3f}s"
        )
    final_midi_bpm = float(midi_bundle_data.render_bpm) * float(sync_info["tempo_factor"])
    midi_zip_path = export_midi_stem_bundle(
        midi_bundle_data,
        output_zip_path=build_midi_stem_zip_path(final_output),
        tempo_bpm=final_midi_bpm,
    )
    print(f"{tag}MIDI stems ZIP: {midi_zip_path} | tempo={final_midi_bpm:.3f} BPM", flush=True)
    print(f"{tag}Done -> {final_output}", flush=True)


def main() -> None:
    args = parse_args()

    if args.batch:
        input_dir = str(Path(args.input).expanduser().resolve())
        output_dir = str(Path(args.output).expanduser().resolve())
        if not os.path.isdir(input_dir):
            raise SystemExit(f"Batch mode requires an input directory: {input_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        jobs = discover_batch_inputs(input_dir)
        if not jobs:
            raise SystemExit("No input-<n> video files found in input directory.")

        failures: List[Tuple[str, str]] = []
        total = len(jobs)
        for i, (num, in_path) in enumerate(jobs, start=1):
            out_path = str(Path(output_dir) / f"output-{num}.wav")
            label = f"{i}/{total}"
            try:
                run_single_pipeline(args, input_path=in_path, output_path=out_path, label=label)
            except Exception as exc:
                failures.append((in_path, str(exc)))
                print(f"[FAILED] {Path(in_path).name}: {exc}", flush=True)

        if failures:
            msg = "; ".join(f"{Path(p).name}: {err}" for p, err in failures[:5])
            raise SystemExit(f"Batch completed with {len(failures)} failures. {msg}")
        print(f"Batch completed: {total} files -> {output_dir}")
        return

    input_path = str(Path(args.input).expanduser().resolve())
    if not is_video(input_path):
        raise SystemExit(f"Input must be a video file: {input_path}")
    output_path = str(Path(args.output).expanduser().resolve())
    run_single_pipeline(args, input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()

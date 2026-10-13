#!/usr/bin/env python3
"""Color-to-Music Composition System
- HSV 色彩空间
- 低维特征提取（K-means 或色相直方图峰值）
- 光学频率对数压缩参数（仅用于调制音乐决策）
- 音乐映射（量化到音阶、和声规则、节奏规则）
- MIDI 输出（和声/低音/旋律/纹理）
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# [EN] Runtime compatibility: suppress loky physical-core warnings observed on some macOS builds.
# [中文] 运行时兼容性：抑制部分 macOS 环境下出现的 loky 物理核心告警。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from simple_kmeans import KMeans
_KMEANS_ERR = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import mido


C_M_PER_S = 3e8
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
INPUT_BATCH_RE = re.compile(r"^input-(\d+)\.(mp4|mov|mkv|avi|webm)$", re.IGNORECASE)
HUE_OCTAVES = 5
HUE_OCTAVE_START = 2
MELODY_RANGE = (60, 84)   # [EN] C4-C6. [中文] C4-C6。
HARMONY_RANGE = (36, 60)  # [EN] C2-C4. [中文] C2-C4。
BASS_RANGE = (36, 48)     # [EN] C2-C3. [中文] C2-C3。
PAD_RANGE = (48, 72)      # [EN] C3-C5. [中文] C3-C5。

#情绪预设(可选，默认不用)
MOODS = {
    # [EN] Bright mood preset: conservative harmony and smaller melodic leaps for a lighter contour.
    # [中文] 明亮预设：采用更保守的和声与更小的旋律跳进，整体线条更轻快。
    "cheerful": {
        "scale": "major",
        "tempo": 150,
        "sync_factor": 0.25,
        "triads_only": True,
        "max_jump": 4,
        "density_boost": 1,
        "tension_factor": 0.6,
        "pad_gain": 0.9,
        "bass_gain": 1.0,
    },
    # [EN] Epic mood preset: denser pad/bass energy with broader harmonic color.
    # [中文] 史诗预设：铺底与低音能量更厚，和声色彩范围更宽。
    "epic": {
        "scale": "mixolydian",
        "tempo": 120,
        "sync_factor": 0.2,
        "triads_only": False,
        "max_jump": 5,
        "density_boost": 0,
        "tension_factor": 0.9,
        "pad_gain": 1.2,
        "bass_gain": 1.2,
        "add_pad_octave": True,
    },
}

@dataclass(frozen=True)
class SegmentInfo:
    features: SegmentFeatures
    start: int
    end: int


@dataclass(frozen=True)
class SegmentModulePlan:
    start_tick: int
    end_tick: int
    harmony_notes: Tuple[int, ...]
    harmony_velocity: int
    bass_note: int
    bass_velocity: int
    pad_notes: Tuple[int, ...]
    pad_velocity: int
    color_span: float = 0.0


def f_param_from_wavelength_nm(wavelength_nm: float) -> float:
    """EN: Convert wavelength (nm) to the project's log optical-frequency parameter.
    中文：将波长（纳米）转换为项目使用的“光频对数参数”。

    EN: Raw light frequency values are too large for direct musical mapping.
    A log transform keeps values numerically stable and musically usable.
    中文：真实光频数值过大，不适合直接映射音乐；对数压缩后更稳定、更易用于调制。
    """
    wavelength_m = wavelength_nm * 1e-9
    f_light = C_M_PER_S / wavelength_m
    return math.log(f_light / 1e4)


F_PARAM_MIN = f_param_from_wavelength_nm(750.0)
F_PARAM_MAX = f_param_from_wavelength_nm(380.0)
F_PARAM_RANGE = max(1e-6, F_PARAM_MAX - F_PARAM_MIN)

@dataclass(frozen=True)
class ColorCluster:
    hue: float  # [EN] Hue angle in degrees [0, 360). [中文] 色相角度[0, 360)。
    sat: float  # [EN] Saturation in normalized range [0, 1]. [中文] 饱和度归一化到 [0, 1]。
    val: float  # [EN] Brightness/value in normalized range [0, 1]. [中文] 明度（Value）归一化到 [0, 1]。
    weight: float  # [EN] Cluster weight in normalized range [0, 1]. [中文] 聚类权重归一化到 [0, 1]。
    f_param: float  # [EN] Log-compressed optical-frequency descriptor. [中文] 对数压缩后的光频描述参数。


@dataclass(frozen=True)
class SegmentFeatures:
    primary: ColorCluster
    secondary: Optional[ColorCluster]
    hue_mean: float
    sat_mean: float
    val_mean: float
    hue_var: float
    sat_var: float
    val_var: float
    f_param_mean: float  # [EN] Mean optical-frequency descriptor for this segment. [中文] 该时间段的光频描述符均值。
    f_param_var: float  # [EN] Variance of optical descriptor; indicates intra-segment visual instability. [中文] 光频描述符方差，表示段内视觉不稳定度。


# [EN] Scale dictionary: semitone offsets from tonic (pitch-class domain, one octave).
# [中文] 调式字典：以主音为基准的半音偏移（音级域，单八度）。
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],  # [EN] Ionian mode. [中文] 自然大调（Ionian）。
    "minor": [0, 2, 3, 5, 7, 8, 10],  # [EN] Natural minor (Aeolian). [中文] 自然小调（Aeolian）。
    "dorian": [0, 2, 3, 5, 7, 9, 10],  # [EN] Dorian mode. [中文] 多利亚调式。
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],  # [EN] Mixolydian mode (major with b7). [中文] 混合利底亚调式（大调降七）。
    "pentatonic": [0, 2, 4, 7, 9],  # [EN] Major pentatonic scale. [中文] 大调五声音阶。
}


# ============================================================================
# [EN] CLI Configuration
# [中文] 命令行配置
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Purpose / 目的
    EN: Define command-line options for feature extraction, composition, and rendering.
    中文：定义特征提取、作曲映射与渲染输出所需的命令行参数。

    Outputs / 输出
    EN: Parsed `argparse.Namespace` consumed by `main()`.
    中文：返回 `argparse.Namespace`，供 `main()` 调度全流程。

    Notes / 说明
    EN: Output mode is mutually exclusive:
    - MIDI export (default)
    - `--direct-output` (SoundFont + fluidsynth)
    - `--synth-output` (built-in oscillator/additive renderer)
    中文：输出模式互斥：
    - 默认导出 MIDI
    - `--direct-output`（SoundFont + fluidsynth）
    - `--synth-output`（内置振荡器/加法合成）
    """
    parser = argparse.ArgumentParser(description="Generate MIDI from video using perceptual color mapping.")
    # [EN] Positional input path: source video file.
    # [中文] 位置参数：输入视频文件路径。
    parser.add_argument("input", help="Path to video file (or input directory when --batch is set).")
    # [EN] Positional output path: target MIDI/WAV path (depends on render mode).
    # [中文] 位置参数：输出路径（具体为 MIDI 或 WAV 取决于渲染模式）。
    parser.add_argument("output", help="Path to output file (or output directory when --batch is set).")
    # [EN] Batch mode: treat input/output as directories and process input-<n>.mp4 sequentially.
    # [中文] 批处理模式：将 input/output 视为目录，顺序处理 input-<n>.mp4。
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: input/output are directories; convert input-<n>.mp4 to output-<n>.wav sequentially.",
    )

    # [EN] Number of dominant color clusters used by K-Means.
    # [中文] K-Means 主色聚类数量。
    parser.add_argument("--k", type=int, default=5, help="K-means clusters (3-7 recommended).")
    # [EN] Feature reduction backend: kmeans or hue histogram.
    # [中文] 特征降维后端：kmeans 或色相直方图。
    parser.add_argument("--method", choices=["kmeans", "hist"], default="kmeans", help="Feature reduction method.")
    # [EN] Hue histogram bins when `--method hist` is selected.
    # [中文] 选择 `--method hist` 时的色相直方图分箱数。
    parser.add_argument("--bins", type=int, default=36, help="Hue histogram bins (for --method hist).")

    # [EN] Resize longest frame side to bound processing cost.
    # [中文] 将帧长边缩放到该值以控制处理开销。
    parser.add_argument("--max-size", type=int, default=512, help="Resize longest side for speed.")
    # [EN] Maximum sampled pixels per segment for statistics.
    # [中文] 每个分段用于统计的最大采样像素数。
    parser.add_argument("--sample", type=int, default=60000, help="Max pixels sampled per segment.")

    # [EN] Minimum frame granularity of segment construction in video mode.
    # [中文] 视频模式下分段构建的基础帧粒度。
    parser.add_argument("--frames-per-segment", type=int, default=5, help="N frames per musical segment.")
    # [EN] Decode stride: read every Nth frame.
    # [中文] 解码步长：每 N 帧读取一帧。
    parser.add_argument("--frame-step", type=int, default=1, help="Frame stride while reading video.")
    # [EN] Hard cap on decoded frame count.
    # [中文] 解码帧数量上限。
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to read.")

    # [EN] Smoothing window size used by S(t) and related temporal curves.
    # [中文] S(t) 及相关时序曲线的平滑窗口大小。
    parser.add_argument("--smooth", type=int, default=3, help="Moving average window (odd).")

    # [EN] Optional fixed tempo override in BPM.
    # [中文] 可选固定速度（BPM）覆盖自动估计。
    parser.add_argument("--tempo", type=int, default=None, help="Tempo (BPM). If omitted, auto rhythm is used.")
    # [EN] Scale factor for segment duration on musical timeline.
    # [中文] 分段在音乐时间线上的时长缩放系数。
    parser.add_argument("--bars-per-segment", type=int, default=1, help="Scale segment duration (default 1x).")
    # [EN] Force scale/mode and skip automatic scale selection.
    # [中文] 强制指定调式，跳过自动调式选择。
    parser.add_argument("--scale", choices=sorted(SCALES.keys()), help="Force scale/mode.")
    # [EN] Base random seed for reproducible stochastic steps.
    # [中文] 随机步骤的基础种子（用于可复现）。
    parser.add_argument("--seed", type=int, default=7, help="Random seed (sampling only).")
    # [EN] High-level mood preset affecting tempo/harmony behavior.
    # [中文] 高层情绪预设，影响速度与和声策略。
    parser.add_argument("--mood", choices=sorted(MOODS.keys()), help="Mood preset (cheerful/epic).")
    # [EN] Sensitivity gain for local visual change response.
    # [中文] 局部视觉变化响应灵敏度增益。
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Increase response to small visual changes.")
    # [EN] Blend between safer post-fallback control and pure visual mapping behavior.
    # [中文] 在更保守的后处理回退与更纯粹的视觉映射之间进行权重融合。
    parser.add_argument(
        "--mapping-bias",
        type=float,
        default=0.5,
        help="Safety fallback blend (0.0=safety correction, 1.0=pure visual mapping).",
    )
    render_group = parser.add_mutually_exclusive_group()
    # [EN] Render directly to WAV via fluidsynth using SoundFont.
    # [中文] 使用 SoundFont + fluidsynth 直接渲染为 WAV。
    render_group.add_argument(
        "--direct-output",
        action="store_true",
        help="Render directly to a WAV audio file using a SoundFont instrument library (via fluidsynth).",
    )
    # [EN] Render directly to WAV via built-in oscillator/additive synth.
    # [中文] 使用内置振荡器/加法合成器直接渲染为 WAV。
    render_group.add_argument(
        "--synth-output",
        action="store_true",
        help="Render directly to a WAV audio file using the built-in synth (ADSR + subtle reverb).",
    )
    # [EN] Path to `.sf2` library (required when `--direct-output` is on).
    # [中文] `.sf2` 音色库路径（开启 `--direct-output` 时必填）。
    parser.add_argument("--soundfont", type=str, default=None, help="Path to a .sf2 SoundFont (required for --direct-output).")
    # [EN] Output sample rate for WAV render modes.
    # [中文] WAV 渲染模式下的输出采样率。
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate for audio rendering modes.")
    # [EN] Global fluidsynth gain in SoundFont rendering mode.
    # [中文] SoundFont 渲染模式下的 fluidsynth 全局增益。
    parser.add_argument("--fluidsynth-gain", type=float, default=0.8, help="fluidsynth gain (0.0-5.0).")
    # [EN] Force single-channel MIDI (all parts to channel 1).
    # [中文] 强制单通道 MIDI（所有声部分配到通道 1）。
    parser.add_argument(
        "--single-channel",
        dest="single_channel",
        action="store_true",
        help="Force all tracks to MIDI channel 1 (default).",
    )
    # [EN] Use separate MIDI channels for Harmony/Bass/Melody/Pad.
    # [中文] 为和声/低音/旋律/铺底使用独立 MIDI 通道。
    parser.add_argument(
        "--multi-channel",
        dest="single_channel",
        action="store_false",
        help="Use separate MIDI channels per track.",
    )
    # [EN] Enable Logic Pro-friendly metadata and multitrack behavior.
    # [中文] 启用 Logic Pro 友好的元数据与多轨行为。
    parser.add_argument(
        "--logic-pro",
        action="store_true",
        help="Logic Pro mode: force multi-track Type 1 and write track/instrument names.",
    )
    # [EN] Insert GM initialization/program setup for auto instrument assignment.
    # [中文] 写入 GM 初始化与音色设置，实现自动音色分配。
    parser.add_argument(
        "--auto-instrument",
        action="store_true",
        help="Write GM reset + bank select + program changes so GM multi-timbral instruments auto-assign sounds.",
    )
    # [EN] Export DAW-friendly folder structure with per-track MIDI files.
    # [中文] 导出 DAW 友好的分轨 MIDI 文件夹结构。
    parser.add_argument(
        "--logic-structure",
        action="store_true",
        help="Export Logic-friendly folder with separate MIDI files per track.",
    )

    # [EN] Debug-only: treat image strips as pseudo-time sequence.
    # [中文] 仅调试：将图像条带视作伪时间序列。
    parser.add_argument("--line-scan", action="store_true", help="Treat image rows as time slices (debug).")
    # [EN] Number of strips used in line-scan debug mode.
    # [中文] 行扫描调试模式下的条带数量。
    parser.add_argument("--slices", type=int, default=16, help="Line-scan slices.")

    parser.set_defaults(single_channel=True)
    return parser.parse_args()


# ============================================================================
# [EN] Visual Processing
# [中文] 视觉预处理
# ============================================================================

def load_image(path: str, max_size: int) -> Image.Image:
    """EN: Load image file and normalize to RGB before HSV conversion.
    中文：读取图像并统一转为 RGB，供后续 HSV 特征提取。"""
    img = Image.open(path).convert("RGB")
    return resize_pil(img, max_size)


def resize_pil(img: Image.Image, max_size: int) -> Image.Image:
    """EN: Resize while preserving aspect ratio; clamp longest side to max_size.
    中文：保持宽高比缩放，限制长边不超过 max_size。"""
    w, h = img.size
    scale = max(w, h) / max_size if max(w, h) > max_size else 1.0
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    return img


def rgb_to_hsv_array(img: Image.Image) -> np.ndarray:
    """EN: Convert PIL RGB to float HSV array used by the mapping pipeline.
    中文：将 PIL RGB 转为浮点 HSV，作为视觉到音乐映射的统一输入格式。

    EN ranges: H in [0,360), S in [0,1], V in [0,1].
    中文范围：H 在 [0,360)，S/V 在 [0,1]。
    """
    hsv = img.convert("HSV")
    arr = np.asarray(hsv).astype(np.float32)
    # [EN] PIL stores HSV as integers in [0, 255]; convert to normalized floats before analysis.
    # [中文] PIL 的 HSV 为 [0,255] 整数域；分析前先转换为归一化浮点域。
    arr[..., 0] = (arr[..., 0] / 255.0) * 360.0
    arr[..., 1] = arr[..., 1] / 255.0
    arr[..., 2] = arr[..., 2] / 255.0
    return arr


def load_video_frames(
    path: str,
    max_size: int,
    frame_step: int,
    max_frames: int,
    progress_desc: Optional[str] = None,
) -> Tuple[List[np.ndarray], float]:
    """Purpose / 目的
    EN: Decode a video file into an HSV frame timeline for downstream analysis.
    中文：将视频解码为 HSV 帧时间序列，供后续视觉特征与节奏计算。

    Inputs / 输入
    EN: `path`, spatial cap `max_size`, temporal sampling `frame_step`, frame limit `max_frames`.
    中文：视频路径、长边上限 `max_size`、抽帧步长 `frame_step`、最大帧数 `max_frames`。

    Returns / 输出
    EN: `(frames, fps)` where `frames` is a list of HSV float arrays.
    中文：返回 `(frames, fps)`，其中 `frames` 为 HSV 浮点数组列表。

    Method / 方法
    EN: Read metadata -> estimate FPS -> subsample frames -> resize -> RGB->HSV conversion.
    中文：读取元数据 -> 估计 FPS -> 抽帧 -> 缩放 -> RGB 转 HSV。
    """
    if imageio is None:
        raise RuntimeError("Video input requires imageio (pip install imageio).")

    try:
        reader = imageio.get_reader(path)
    except ValueError as exc:
        raise RuntimeError(
            "无法打开视频。请安装解码后端：pip install \"imageio[ffmpeg]\" "
            "（或 pip install \"imageio[pyav]\"）。"
        ) from exc
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 0.0)
    if fps <= 0 and meta.get("duration") and meta.get("nframes"):
        try:
            fps = float(meta["nframes"]) / float(meta["duration"])
        except Exception:
            fps = 0.0
    if fps <= 0:
        fps = 30.0
    frames: List[np.ndarray] = []
    stride = max(1, int(frame_step))

    nframes_raw = meta.get("nframes")
    nframes_total: Optional[int] = None
    if isinstance(nframes_raw, (int, float)):
        try:
            n_est = int(nframes_raw)
            if n_est > 0 and n_est < 10_000_000:
                nframes_total = n_est
        except Exception:
            nframes_total = None

    bar = None
    use_bar = (tqdm is not None) and bool(progress_desc)
    if use_bar:
        total_for_bar: Optional[int] = nframes_total
        if total_for_bar is not None and max_frames:
            total_for_bar = min(total_for_bar, max(1, int(max_frames)) * stride)
        if nframes_total is not None:
            bar = tqdm(total=total_for_bar, desc=str(progress_desc), unit="frame", leave=True, file=sys.stdout)
        else:
            bar = tqdm(desc=str(progress_desc), unit="frame", leave=True, file=sys.stdout)

    try:
        for i, frame in enumerate(reader):
            if bar is not None:
                bar.update(1)
            if i % stride != 0:
                continue
            img = Image.fromarray(frame).convert("RGB")
            img = resize_pil(img, max_size)
            frames.append(rgb_to_hsv_array(img))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        if bar is not None:
            bar.close()
    reader.close()
    return frames, fps


# ============================================================================
# [EN] Feature Reduction
# [中文] 特征降维与聚合
# ============================================================================

def hue_to_wavelength_nm(hue: float) -> float:
    """EN: Map hue angle to a conceptual visible-light wavelength.
    中文：将色相角映射到“概念化”的可见光波长。

    EN: This is an artistic mapping used for modulation, not a strict physics model.
    中文：这是艺术化映射，用于调制，不是严格物理反演。
    """
    # [EN] Conceptual hue-to-wavelength mapping used as an artistic modulation bridge.
    # [中文] 采用概念化“色相到波长”映射，作为艺术化调制桥梁。
    hue = hue % 360.0
    return 750.0 - (hue / 360.0) * 370.0


def optical_frequency_param(hue: float) -> float:
    """EN: Compute the log optical-frequency parameter from hue.
    中文：从色相计算光频对数参数。"""
    wavelength_nm = hue_to_wavelength_nm(hue)
    wavelength_m = wavelength_nm * 1e-9
    f_light = C_M_PER_S / wavelength_m
    return math.log(f_light / 1e4)


def sample_pixels(hsv: np.ndarray, sample: int, seed: int) -> np.ndarray:
    """EN: Randomly subsample pixels to reduce compute while preserving distribution.
    中文：随机抽样像素，降低计算量，同时尽量保持颜色统计分布。"""
    pixels = hsv.reshape(-1, 3)
    if sample and pixels.shape[0] > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pixels.shape[0], size=sample, replace=False)
        pixels = pixels[idx]
    return pixels


def compute_clusters_kmeans(pixels: np.ndarray, k: int, seed: int) -> List[ColorCluster]:
    """Purpose / 目的
    EN: Extract dominant color groups from sampled HSV pixels using K-Means.
    中文：在 HSV 采样像素上使用 K-Means 提取主导颜色簇。

    Outputs / 输出
    EN: A sorted list of `ColorCluster` with center hue/sat/val, weight, and optical parameter.
    中文：返回按权重排序的 `ColorCluster` 列表，包含中心色、权重与光频参数。

    Notes / 说明
    EN: This path is more accurate than histogram peak extraction but typically slower.
    中文：相较直方图法更精细，但计算开销更高。
    """
    if KMeans is None:
        raise RuntimeError("scikit-learn is required for k-means clustering.") from _KMEANS_ERR

    k = max(1, k)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(pixels)

    clusters: List[ColorCluster] = []
    total = pixels.shape[0]
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue
        subset = pixels[mask]
        hue, sat, val = subset.mean(axis=0)
        weight = subset.shape[0] / total
        clusters.append(ColorCluster(hue, sat, val, weight, optical_frequency_param(hue)))

    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


def compute_clusters_histogram(pixels: np.ndarray, bins: int, k: int) -> List[ColorCluster]:
    """Purpose / 目的
    EN: Fast dominant-color extraction by selecting peaks in hue histogram.
    中文：通过色相直方图峰值快速提取主导颜色。

    Outputs / 输出
    EN: `ColorCluster` list derived from top histogram bins.
    中文：由直方图主峰构造的 `ColorCluster` 列表。

    Notes / 说明
    EN: Preferred for speed-critical runs; less expressive than K-Means in mixed-color scenes.
    中文：适合速度优先场景；在复杂混色画面上表达能力弱于 K-Means。
    """
    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]

    hist, bin_edges = np.histogram(hue, bins=bins, range=(0.0, 360.0))
    peak_bins = np.argsort(hist)[::-1][: max(1, k)]

    clusters: List[ColorCluster] = []
    total = pixels.shape[0]
    for b in peak_bins:
        low = bin_edges[b]
        high = bin_edges[b + 1]
        mask = (hue >= low) & (hue < high)
        if not np.any(mask):
            continue
        h = float(hue[mask].mean())
        s = float(sat[mask].mean())
        v = float(val[mask].mean())
        weight = float(np.sum(mask) / total)
        clusters.append(ColorCluster(h, s, v, weight, optical_frequency_param(h)))

    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


def build_segment_features(clusters: Sequence[ColorCluster]) -> SegmentFeatures:
    """Purpose / 目的
    EN: Convert cluster descriptors into one segment-level feature object.
    中文：将颜色簇描述聚合为单个“时间段特征”对象。

    Outputs / 输出
    EN: `SegmentFeatures` including weighted circular hue stats, sat/val stats,
    and optical parameter statistics.
    中文：返回 `SegmentFeatures`，包含色相环形统计、饱和/亮度统计与光频统计。

    Why / 原因
    EN: Segment-level features are the bridge from visual analysis to music mapping.
    中文：分段特征是“视觉分析 -> 音乐映射”的核心桥梁。
    """
    clusters = list(clusters)
    clusters.sort(key=lambda c: c.weight, reverse=True)
    primary = clusters[0]
    secondary = clusters[1] if len(clusters) > 1 else None

    hues = [c.hue for c in clusters]
    weights = [c.weight for c in clusters]
    hue_mean = weighted_circular_mean_deg(hues, weights)
    hue_var = weighted_circular_var_deg(hues, weights)
    sat_mean = float(np.average([c.sat for c in clusters], weights=weights))
    val_mean = float(np.average([c.val for c in clusters], weights=weights))
    sat_var = float(np.average([(c.sat - sat_mean) ** 2 for c in clusters], weights=weights))
    val_var = float(np.average([(c.val - val_mean) ** 2 for c in clusters], weights=weights))
    f_param_vals = [c.f_param for c in clusters]
    f_param_mean = float(np.average(f_param_vals, weights=weights))
    f_param_var = float(np.average([(v - f_param_mean) ** 2 for v in f_param_vals], weights=weights))

    return SegmentFeatures(
        primary=primary,
        secondary=secondary,
        hue_mean=hue_mean,
        sat_mean=sat_mean,
        val_mean=val_mean,
        hue_var=hue_var,
        sat_var=sat_var,
        val_var=val_var,
        f_param_mean=f_param_mean,
        f_param_var=f_param_var,
    )


def segment_video_frames(frames: List[np.ndarray], frames_per_segment: int, method: str, k: int, bins: int, sample: int, seed: int) -> List[SegmentFeatures]:
    """Purpose / 目的
    EN: Build segment features with a fixed frame window, mainly for legacy/debug paths.
    中文：按固定帧窗口构建分段特征，主要用于旧流程兼容与调试。

    Inputs / 输入
    EN:
    - `frames`: HSV frame list
    - `frames_per_segment`: fixed segment width in frames
    - `method`: `kmeans` or `hist` dominant-color reduction
    - `k`, `bins`, `sample`, `seed`: clustering and sampling controls
    中文：
    - `frames`：HSV 帧序列
    - `frames_per_segment`：固定分段长度（帧）
    - `method`：`kmeans` 或 `hist` 主色提取
    - `k`、`bins`、`sample`、`seed`：聚类与采样参数

    Output / 输出
    EN: Ordered `SegmentFeatures` list aligned with fixed windows.
    中文：与固定窗口顺序对齐的 `SegmentFeatures` 列表。
    """
    frames_per_segment = max(1, frames_per_segment)
    segments: List[SegmentFeatures] = []

    for i in range(0, len(frames), frames_per_segment):
        block = frames[i : i + frames_per_segment]
        if not block:
            continue
        merged = np.concatenate([f.reshape(-1, 3) for f in block], axis=0)
        pixels = sample_pixels(merged.reshape(-1, 3), sample, seed + i)

        if method == "kmeans":
            clusters = compute_clusters_kmeans(pixels, k, seed + i)
        else:
            clusters = compute_clusters_histogram(pixels, bins, k)

        segments.append(build_segment_features(clusters))

    return segments


def split_hsv_slices(hsv: np.ndarray, slices: int) -> List[np.ndarray]:
    """Purpose / 目的
    EN: Split one HSV image into horizontal strips used as pseudo-time slices.
    中文：将单张 HSV 图像按水平方向切成条带，作为伪时间切片。

    Notes / 说明
    EN: This is only for `--line-scan` debug mode and is not used in normal video flow.
    中文：该函数仅用于 `--line-scan` 调试模式，不参与正常视频流程。
    """
    h, _, _ = hsv.shape
    slices = max(1, slices)
    step = h / slices
    strips: List[np.ndarray] = []
    for i in range(slices):
        start = int(i * step)
        end = int((i + 1) * step)
        if end <= start:
            end = min(h, start + 1)
        strips.append(hsv[start:end, :, :])
    return strips


def compute_line_scan_segments(hsv: np.ndarray, slices: int) -> Tuple[List[SegmentFeatures], List[np.ndarray]]:
    """Purpose / 目的
    EN: Build pseudo temporal segments from strip means in line-scan mode.
    中文：在行扫描模式下，根据条带均值构建伪时间分段特征。

    Method / 方法
    EN:
    1) split image into strips
    2) treat each strip mean HSV as one lightweight color cluster
    3) lift to `SegmentFeatures`
    中文：
    1）图像切条带
    2）每条带均值 HSV 视作一个简化颜色簇
    3）转换为 `SegmentFeatures`
    """
    strips = split_hsv_slices(hsv, slices)
    segments: List[SegmentFeatures] = []
    for strip in strips:
        hue, sat, val = strip.reshape(-1, 3).mean(axis=0)
        weight = 1.0 / max(1, len(strips))
        cluster = ColorCluster(hue, sat, val, weight, optical_frequency_param(hue))
        segments.append(build_segment_features([cluster]))
    return segments, strips


def circular_mean_deg(values: Sequence[float]) -> float:
    """Purpose / 目的
    EN: Compute wrap-safe angular mean in degrees (for hue-like variables).
    中文：计算角度的环形均值（适用于色相等周期变量）。

    Why / 原因
    EN: Linear mean fails near 0/360 boundary (e.g., mean of 359 and 1 should be 0, not 180).
    中文：线性均值在 0/360 边界会失真（如 359 与 1 的均值应接近 0，而不是 180）。
    """
    if len(values) == 0:
        return 0.0
    radians = np.deg2rad(values)
    s = float(np.mean(np.sin(radians)))
    c = float(np.mean(np.cos(radians)))
    ang = math.degrees(math.atan2(s, c))
    return ang % 360.0


def weighted_circular_mean_deg(values: Sequence[float], weights: Sequence[float]) -> float:
    """Purpose / 目的
    EN: Weighted circular mean in degrees, emphasizing dominant-color clusters.
    中文：带权重的环形均值（度），用于强调主导颜色簇。

    Notes / 说明
    EN: Weights are normalized internally; zero-sum weights fall back safely.
    中文：函数内部会归一化权重；权重和为零时有安全回退。
    """
    if len(values) == 0:
        return 0.0
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
    radians = np.deg2rad(values)
    s = float(np.sum(np.sin(radians) * weights))
    c = float(np.sum(np.cos(radians) * weights))
    ang = math.degrees(math.atan2(s, c))
    return ang % 360.0


def weighted_circular_var_deg(values: Sequence[float], weights: Sequence[float]) -> float:
    """Purpose / 目的
    EN: Weighted circular variance in [0,1] to measure hue dispersion.
    中文：计算 [0,1] 范围的带权环形方差，用于衡量色相离散度。

    Interpretation / 解释
    EN:
    - near 0: hues are concentrated (stable color center)
    - near 1: hues are widely spread (high chromatic diversity)
    中文：
    - 接近 0：色相集中（颜色中心稳定）
    - 接近 1：色相分散（颜色多样性高）
    """
    if len(values) == 0:
        return 0.0
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
    radians = np.deg2rad(values)
    s = float(np.sum(np.sin(radians) * weights))
    c = float(np.sum(np.cos(radians) * weights))
    r = math.sqrt(s * s + c * c)
    return float(max(0.0, min(1.0, 1.0 - r)))


def smooth_segments(segments: List[SegmentFeatures], window: int) -> List[SegmentFeatures]:
    """EN: Smooth segment features over time to reduce abrupt harmony jumps.
    中文：在时间维度平滑段特征，减少和声与旋律的突兀跳变。

    EN:
    - circular-mean smoothing for hue
    - arithmetic-mean smoothing for sat/val and variances
    - keep `primary/secondary` identity from center segment for interpretability
    中文：
    - 色相使用环形均值平滑
    - 饱和度/亮度及方差使用算术均值平滑
    - `primary/secondary` 保留中心段身份，便于解释与追踪
    """
    if window <= 1 or len(segments) < 3:
        return segments
    if window % 2 == 0:
        window += 1
    half = window // 2
    out: List[SegmentFeatures] = []
    for i in range(len(segments)):
        lo = max(0, i - half)
        hi = min(len(segments), i + half + 1)
        subset = segments[lo:hi]
        # [EN] Hue must be smoothed on a circle to avoid 0/360 discontinuity artifacts.
        # [中文] 色相必须在环形空间平滑，避免 0/360 边界伪跳变。
        hue = circular_mean_deg([c.hue_mean for c in subset])
        sat = float(np.mean([c.sat_mean for c in subset]))
        val = float(np.mean([c.val_mean for c in subset]))
        hue_var = float(np.mean([c.hue_var for c in subset]))
        sat_var = float(np.mean([c.sat_var for c in subset]))
        val_var = float(np.mean([c.val_var for c in subset]))
        f_param_var = float(np.mean([c.f_param_var for c in subset]))
        primary = segments[i].primary
        secondary = segments[i].secondary
        f_param_mean = optical_frequency_param(hue)
        out.append(
            SegmentFeatures(
                primary=primary,
                secondary=secondary,
                hue_mean=hue,
                sat_mean=sat,
                val_mean=val,
                hue_var=hue_var,
                sat_var=sat_var,
                val_var=val_var,
                f_param_mean=f_param_mean,
                f_param_var=f_param_var,
            )
        )
    return out


# ============================================================================
# [EN] Musical Mapping
# [中文] 音乐映射
# ============================================================================

def pitch_from_hue(hue: float) -> Tuple[int, int]:
    """EN: Map hue to (pitch-class, octave) within a fixed 5-octave span.
    中文：把色相映射为（音级 pitch class, 八度），范围固定为 5 个八度。

    EN mapping:
    - hue in [0, 360) -> semitone index in [0, 60)
    - semitone index -> `(pc, octave)`
    中文映射：
    - 色相 [0,360) -> 半音索引 [0,60)
    - 半音索引 -> `(pc, octave)`
    """
    # [EN] Map the full hue circle to a five-octave pitch span (60 semitones).
    # [中文] 将完整色环映射到五个八度（共 60 个半音）。
    hue = hue % 360.0
    semitone = int((hue / 360.0) * (HUE_OCTAVES * 12))
    semitone = min(HUE_OCTAVES * 12 - 1, max(0, semitone))
    octave = HUE_OCTAVE_START + (semitone // 12)
    pc = semitone % 12
    return pc, octave


def octave_from_value(val: float) -> int:
    """EN: Experimental brightness-to-octave helper.
    中文：亮度到八度的实验性辅助函数。

    EN: Currently kept for experimentation and not used as a hard rule in main melody path.
    中文：当前主要用于实验，不作为主旋律路径的硬约束规则。
    """
    return int(round(2 + val * 4))


def velocity_from_value(val: float) -> int:
    """EN: Map brightness to MIDI velocity using log compression.
    中文：用对数压缩将亮度映射到 MIDI 力度。

    EN: Log mapping avoids overly aggressive velocity growth at high brightness.
    中文：对数映射可避免高亮区域力度增长过快。
    """
    return int(max(1, min(127, math.log(val + 1.0) * 127)))


def optical_trend(segments: Sequence[SegmentFeatures]) -> float:
    """EN: Normalize optical parameter trend within per-video min/max range.
    中文：将光频参数趋势归一化到该视频自身范围内。

    EN: Returns where the mean optical descriptor sits between segment-wise min and max.
    中文：返回光频均值在“该视频最小-最大”区间中的相对位置。
    """
    vals = [c.f_param_mean for c in segments]
    if len(vals) < 2:
        return 0.5
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return 0.5
    mean = float(np.mean(vals))
    return (mean - lo) / (hi - lo)


def choose_scale(segments: Sequence[SegmentFeatures], override: Optional[str] = None) -> str:
    """Purpose / 目的
    EN: Choose a musical scale from segment statistics unless manually overridden.
    中文：在没有手动覆盖时，根据分段统计自动选择调式。

    Inputs / 输入
    EN: Segment-level saturation, brightness, hue trend, and optical trend.
    中文：分段层面的饱和度、亮度、色相趋势与光频趋势。

    Method / 方法
    EN:
    1) low saturation -> pentatonic
    2) hue zone -> major / dorian / minor base
    3) brightness and optical trend adjust final scale
    中文：
    1）低饱和度优先五声音阶
    2）按色相区间得到大调/多利亚/小调基底
    3）再用亮度与光频趋势微调最终结果
    """
    if override:
        return override

    avg_sat = float(np.mean([c.sat_mean for c in segments]))
    avg_val = float(np.mean([c.val_mean for c in segments])) if segments else 0.5
    if avg_sat < 0.3:
        return "pentatonic"

    avg_hue = circular_mean_deg([c.hue_mean for c in segments])
    if avg_hue < 60 or avg_hue >= 300:
        base = "major"
    elif avg_hue < 180:
        base = "dorian"
    else:
        base = "minor"

    # [EN] High brightness biases mode selection toward brighter tonal color.
    # [中文] 高亮画面会将调式选择偏向更明亮的音色倾向。
    if avg_val >= 0.60 and base in ("minor", "dorian"):
        base = "major"
    # [EN] Very dark scenes allow a mild shift toward darker modality while preserving stability.
    # [中文] 极暗画面允许轻微向偏暗调式移动，但整体保持稳定。
    if avg_val <= 0.33 and base == "major":
        base = "dorian"

    trend = optical_trend(segments)
    if base == "minor" and trend > 0.66:
        return "mixolydian"
    if base == "major" and trend < 0.33:
        return "dorian"
    return base


def quantize_pc(pc: int, scale: Sequence[int]) -> int:
    """EN: Quantize arbitrary pitch class to nearest degree in current scale.
    中文：把任意音级吸附到当前调式内最近音。

    EN: Distance is circular modulo-12 distance, so wrap-around is handled correctly.
    中文：距离使用 12 平均律环形距离，能正确处理跨八度边界。
    """
    pc = pc % 12
    best = scale[0]
    best_dist = 12
    for degree in scale:
        dist = min((pc - degree) % 12, (degree - pc) % 12)
        if dist < best_dist:
            best_dist = dist
            best = degree
    return best


def scale_degree_index(pc: int, scale: Sequence[int]) -> int:
    """EN: Return nearest degree index in current scale for a pitch class.
    中文：返回某个音级在当前调式中的最近度数索引。"""
    if not scale:
        return 0
    x = int(pc) % 12
    best_idx = 0
    best_dist = 12
    for i, deg in enumerate(scale):
        d = min((x - int(deg)) % 12, (int(deg) - x) % 12)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return int(best_idx)


def smooth_degree_progression(prev_idx: Optional[int], target_idx: int, degree_count: int, max_step: int = 2) -> int:
    """EN: Constrain root motion on scale-degree ring for smoother harmonic progression.
    中文：在度数环上约束根音运动，令和声进行更平滑。"""
    n = int(max(1, degree_count))
    tgt = int(target_idx) % n
    if prev_idx is None:
        return tgt
    prv = int(prev_idx) % n
    cw = (tgt - prv) % n
    ccw = (prv - tgt) % n
    if min(cw, ccw) <= int(max_step):
        return tgt
    if cw < ccw:
        return int((prv + int(max_step)) % n)
    return int((prv - int(max_step)) % n)


def ring_degree_distance(a: int, b: int, degree_count: int) -> int:
    """EN: Circular distance between two scale-degree indices.
    中文：两个调式度数索引在环形空间中的距离。"""
    n = int(max(1, degree_count))
    x = int(a) % n
    y = int(b) % n
    return int(min((x - y) % n, (y - x) % n))


def harmonic_allowed_next_degrees(prev_idx: int, degree_count: int) -> List[int]:
    """EN: In-key harmonic transition graph (classical-biased for heptatonic scales).
    中文：调内和声转移图（七声音阶时偏向常见功能进行）。"""
    n = int(max(1, degree_count))
    p = int(prev_idx) % n
    if n != 7:
        # [EN] Generic fallback: stay / stepwise / third-related.
        # [中文] 通用回退：保持 / 级进 / 三度关系。
        return sorted(set([p, (p + 1) % n, (p - 1) % n, (p + 2) % n, (p - 2) % n]))
    table = {
        0: [0, 3, 4, 5],  # I -> I/IV/V/vi
        1: [1, 4, 6],     # ii -> ii/V/vii
        2: [2, 5],        # iii -> iii/vi
        3: [3, 1, 4, 0],  # IV -> IV/ii/V/I
        4: [4, 0, 5, 1],  # V -> V/I/vi/ii
        5: [5, 1, 3, 4],  # vi -> vi/ii/IV/V
        6: [6, 0, 4],     # vii -> vii/I/V
    }
    return [int(x) % n for x in table.get(p, [p])]


def chord_tones(
    root_pc: int,
    scale: Sequence[int],
    sat: float,
    tension: float,
    triads_only: bool = False,
) -> List[int]:
    """EN: Choose chord tones from saturation and tension controls.
    中文：根据饱和度与张力参数决定和弦音数量与色彩。

    EN:
    - low saturation: root only / sparse harmony
    - medium saturation: triad
    - high saturation + high tension: optional 7th/9th color tones
    中文：
    - 低饱和：主音或稀疏和声；
    - 中饱和：三和弦；
    - 高饱和且高张力：可加入 7/9 色彩音。
    """
    n = int(max(1, len(scale)))
    root_idx = scale_degree_index(root_pc, scale)
    diatonic_root = int(scale[root_idx % n]) % 12
    if sat < 0.3:
        return [diatonic_root]
    # [EN] Build a diatonic chord by scale-degree offsets (1/3/5), keeping harmony strictly in key.
    # [中文] 用度数偏移（1/3/5）构造调内和弦，保证和声严格在当前调内。
    triad = [
        int(scale[(root_idx + 0) % n]) % 12,
        int(scale[(root_idx + 2) % n]) % 12,
        int(scale[(root_idx + 4) % n]) % 12,
    ]
    if sat < 0.6 or triads_only:
        return triad

    # [EN] Keep baseline harmony consonant; add color tones only under high saturation and tension.
    # [中文] 基础和声保持协和，仅在高饱和与高张力时加入色彩音。
    out = list(triad)
    if sat > 0.75 and tension > 0.80:
        out.append(int(scale[(root_idx + 6) % n]) % 12)  # [EN] Optional 7th extension. [中文] 可选七音扩展。
    if tension > 0.92:
        out.append(int(scale[(root_idx + 1) % n]) % 12)  # [EN] Optional 9th extension. [中文] 可选九音扩展。
    return out


def scale_notes(root_pc: int, scale: Sequence[int], octave: int) -> List[int]:
    """EN: Build absolute MIDI notes for a scale anchored at (root_pc, octave).
    中文：根据根音与八度生成该调式的绝对 MIDI 音高列表。

    EN: Output is one-octave degree list used as local melody candidate pool.
    中文：输出为单八度调式音集合，作为局部旋律候选池。
    """
    base = 12 * (octave + 1)
    return [base + (root_pc + deg) % 12 for deg in scale]


def scale_note_near(scale_midi: Sequence[int], index: int, previous: Optional[int]) -> int:
    """EN: Pick a scale note and octave-shift it near previous melody note.
    中文：选择调内音并进行八度折返，使其靠近上一旋律音。

    EN: Candidate set is `{note-12, note, note+12}`; choose nearest to previous note.
    中文：候选集合为 `{note-12, note, note+12}`，选择与上一音最接近者。
    """
    note = scale_midi[index % len(scale_midi)]
    if previous is None:
        return note
    candidates = [note - 12, note, note + 12]
    return min(candidates, key=lambda n: abs(n - previous))


def limit_melody_jump(scale_midi: Sequence[int], index: int, previous: Optional[int], max_semitone_step: int) -> int:
    """EN: Constrain melodic leap size without leaving the active scale.
    中文：在保持调内约束的前提下限制旋律跳进幅度。

    EN strategy:
    1) propose note near previous
    2) if jump exceeds threshold, walk degree index stepwise in opposite direction
    3) stop once jump is acceptable or search budget is exhausted
    中文策略：
    1）先给出靠近上一音的候选
    2）若跳进超阈值，则按级进方向逐步回退索引
    3）满足阈值即返回，否则在搜索预算内尽量逼近
    """
    note = scale_note_near(scale_midi, index, previous)
    if previous is None:
        return note
    if abs(note - previous) <= max_semitone_step:
        return note
    # [EN] Step through scale degrees until the melodic leap constraint is satisfied.
    # [中文] 在调内逐级回退，直到满足最大跳进约束。
    idx = index
    for _ in range(len(scale_midi) * 2):
        if note > previous:
            idx -= 1
        else:
            idx += 1
        note = scale_note_near(scale_midi, idx, previous)
        if abs(note - previous) <= max_semitone_step:
            return note
    return note


# ============================================================================
# [EN] Rhythm Drivers
# [中文] 节奏驱动
# ============================================================================

def brightness_variance(pixels: np.ndarray) -> float:
    """EN: Auxiliary metric: variance of HSV Value channel.
    中文：辅助指标：HSV 亮度通道方差。

    EN: Higher variance implies stronger local contrast complexity.
    中文：方差越高表示局部明暗对比越复杂。
    """
    return float(np.var(pixels[:, 2]))


def edge_density_from_frame(hsv: np.ndarray) -> float:
    """EN: Auxiliary metric: edge-density proxy from Value gradients.
    中文：辅助指标：由亮度梯度近似得到的边缘密度。

    EN: Uses mean absolute horizontal/vertical gradients and maps to [0,1].
    中文：基于水平/垂直绝对梯度均值，并映射到 [0,1] 区间。
    """
    v = hsv[..., 2]
    dx = np.abs(np.diff(v, axis=1))
    dy = np.abs(np.diff(v, axis=0))
    density = float(np.mean(dx) + np.mean(dy))
    return float(min(1.0, density * 3.0))


def normalize(values: Iterable[float]) -> List[float]:
    """EN: Min-max normalize to [0,1], with safe handling for flat signals.
    中文：最小-最大归一化到 [0,1]，并处理全常量序列。

    EN: Flat sequences return all zeros to avoid division-by-zero and fake dynamics.
    中文：常量序列返回全零，避免除零并避免制造虚假动态。
    """
    values = list(values)
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def luminance_entropy(pixels: np.ndarray, bins: int = 32) -> float:
    """EN: Compute normalized luminance entropy as global visual complexity.
    中文：计算归一化亮度熵，作为全局视觉复杂度指标。

    EN:
    - build histogram on Value channel
    - compute Shannon entropy
    - divide by max entropy `log2(bins)` to obtain [0,1] score
    中文：
    - 在亮度通道构建直方图
    - 计算香农熵
    - 再除以 `log2(bins)` 归一化到 [0,1]
    """
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


def dominant_color_count(clusters: Sequence[ColorCluster], min_weight: float = 0.08) -> int:
    """EN: Count clusters above weight threshold (at least 1).
    中文：统计超过权重阈值的主导颜色簇数量（至少返回 1）。

    EN: This is an interpretable proxy for palette richness.
    中文：该值可作为“画面调色盘丰富度”的可解释代理指标。
    """
    if not clusters:
        return 1
    count = sum(1 for c in clusters if c.weight >= min_weight)
    return max(1, count)


def optical_span_norm(clusters: Sequence[ColorCluster]) -> float:
    """EN: Normalized span of optical parameter across dominant clusters.
    中文：主色簇光频参数跨度的归一化值。

    EN: Large span suggests broad chromatic coverage and tends to push faster/more active mapping.
    中文：跨度越大通常表示颜色覆盖更广，映射上更倾向活跃节奏。
    """
    if not clusters:
        return 0.0
    vals = [c.f_param for c in clusters]
    span = max(vals) - min(vals)
    return float(max(0.0, min(1.0, span / F_PARAM_RANGE)))


def symmetry_score(hsv: np.ndarray) -> float:
    """EN: Bilateral symmetry score on brightness (left-right + top-bottom).
    中文：基于亮度的双向对称评分（左右 + 上下）。

    EN:
    - compare left with mirrored right
    - compare top with mirrored bottom
    - convert mean difference to similarity score in [0,1]
    中文：
    - 比较左侧与镜像右侧
    - 比较上侧与镜像下侧
    - 将平均差异转换为 [0,1] 相似度分数
    """
    v = hsv[..., 2]
    h, w = v.shape
    if h < 2 or w < 2:
        return 1.0
    left = v[:, : w // 2]
    right = v[:, w - w // 2 :]
    right = np.flip(right, axis=1)
    diff_lr = float(np.mean(np.abs(left - right))) if left.size and right.size else 0.0

    top = v[: h // 2, :]
    bottom = v[h - h // 2 :, :]
    bottom = np.flip(bottom, axis=0)
    diff_tb = float(np.mean(np.abs(top - bottom))) if top.size and bottom.size else 0.0

    sym = 1.0 - min(1.0, (diff_lr + diff_tb) / 2.0)
    return float(max(0.0, min(1.0, sym)))


def select_bpm(entropy_norm: float, r_opt_norm: float) -> int:
    """Purpose / 目的
    EN: Map global visual complexity to one stable BPM value for the whole piece.
    中文：将全局视觉复杂度映射为整首稳定的 BPM。

    Inputs / 输入
    EN: normalized luminance entropy and normalized optical span.
    中文：亮度熵归一化值与光频跨度归一化值。

    Method / 方法
    EN: Use piecewise tempo bands (slow / medium / fast) with bounded output.
    中文：采用分段速度区间（慢/中/快）并做上下限约束。
    """
    score = (entropy_norm + r_opt_norm) / 2.0
    if entropy_norm < 0.35 and r_opt_norm < 0.35:
        bpm = 60 + 20 * score
    elif entropy_norm < 0.65:
        bpm = 80 + 30 * score
    else:
        bpm = 110 + 30 * score
    return clamp_int(int(round(bpm)), 40, 200)


def _clamp01(x: float) -> float:
    """EN: Clamp scalar to [0,1].
    中文：将标量约束到 [0,1]。"""
    return float(max(0.0, min(1.0, x)))


def hue_warmth_score(hue: float) -> float:
    """EN: Map hue to a warm-vs-cool score in [0,1].
    中文：将色相映射为冷暖分数（[0,1]）。"""
    h = float(hue) % 360.0
    # [EN] Warmth peak around yellow-orange (~45deg), trough around blue (~225deg).
    # [中文] 暖色峰值约在黄橙（~45度），冷色谷值约在蓝色（~225度）。
    return _clamp01(0.5 + 0.5 * math.cos(math.radians(h - 45.0)))


def affect_from_segment(feat: SegmentFeatures) -> Tuple[float, float]:
    """EN: Estimate valence/arousal from segment color descriptors.
    中文：由分段颜色描述估计 valence/arousal（效价/唤醒）。"""
    sat = _clamp01(feat.sat_mean)
    val = _clamp01(feat.val_mean)
    warm = hue_warmth_score(feat.hue_mean)
    motion = _clamp01(0.6 * feat.hue_var + 1.8 * feat.sat_var + 1.8 * feat.val_var)
    # [EN] Bright + saturated + warm tends to pleasant valence.
    # [中文] 高亮度/高饱和/偏暖通常对应更高效价。
    valence = _clamp01(0.52 * val + 0.30 * sat + 0.18 * warm)
    # [EN] Saturation and local visual instability are primary arousal drivers.
    # [中文] 饱和度与局部视觉不稳定性是主要唤醒驱动。
    arousal = _clamp01(0.48 * sat + 0.28 * (1.0 - val) + 0.24 * motion)
    return valence, arousal


def global_affect(segments: Sequence[SegmentInfo]) -> Tuple[float, float]:
    """EN: Duration-weighted global valence/arousal over all segments.
    中文：对全片段按时长加权的全局效价/唤醒估计。"""
    if not segments:
        return 0.5, 0.5
    total_w = 0.0
    val_sum = 0.0
    aro_sum = 0.0
    for seg in segments:
        w = float(max(1, int(seg.end) - int(seg.start)))
        v, a = affect_from_segment(seg.features)
        total_w += w
        val_sum += v * w
        aro_sum += a * w
    if total_w <= 1e-9:
        return 0.5, 0.5
    return _clamp01(val_sum / total_w), _clamp01(aro_sum / total_w)


def select_bpm_affective(arousal: float) -> int:
    """EN: Affective tempo mapping: low arousal -> slow, high arousal -> fast.
    中文：情感速度映射：低唤醒偏慢，高唤醒偏快。"""
    bpm = 55.0 + 95.0 * _clamp01(arousal)
    return clamp_int(int(round(bpm)), 45, 185)


def rhythm_targets_from_affect(valence: float, arousal: float, activity: float) -> Tuple[float, float, float]:
    """EN: Convert affect + activity to density/syncopation/1-f beta targets.
    中文：把效价+唤醒+活动量映射到密度/切分/1-f beta 目标。"""
    v = _clamp01(valence)
    a = _clamp01(arousal)
    u = _clamp01(activity)
    # [EN] Conservative density range to keep phrasing readable and less frantic.
    # [中文] 更保守的密度范围，避免节奏过快过密。
    density = _clamp01(0.10 + 0.42 * a + 0.18 * u)
    sync_base = 0.50 + 0.28 * (a - 0.5) + 0.18 * ((1.0 - v) - 0.5)
    # [EN] Keep syncopation near a moderate zone (avoid extreme complexity).
    # [中文] 把切分控制在中等区域，避免过稀或过乱。
    sync = _clamp01(0.5 + 0.75 * (max(0.22, min(0.78, sync_base)) - 0.5))
    beta = max(0.35, min(1.80, 1.45 - 0.95 * a + 0.35 * (1.0 - v)))
    return density, sync, beta


def _one_over_f_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """EN: Generate normalized 1/f^beta noise via frequency-domain shaping.
    中文：通过频域整形生成归一化 1/f^beta 噪声。"""
    n = int(max(1, n))
    if n == 1:
        return np.array([0.5], dtype=np.float32)
    b = float(max(0.0, min(2.5, beta)))
    freqs = np.fft.rfftfreq(n)
    amp = np.zeros_like(freqs, dtype=np.float32)
    amp[0] = 0.0
    if amp.size > 1:
        amp[1:] = 1.0 / np.maximum(freqs[1:], 1e-6) ** (b * 0.5)
    real = rng.standard_normal(amp.size).astype(np.float32)
    imag = rng.standard_normal(amp.size).astype(np.float32)
    spec = (real + 1j * imag) * amp
    x = np.fft.irfft(spec, n=n).astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.full(n, 0.5, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def normalize_rhythm_steps_per_beat(steps_per_beat: Optional[object]) -> Optional[int]:
    """EN: Normalize rhythm subdivision selector to {1, 2, 4, 6} or None.
    中文：将节奏细分选择标准化为 {1,2,4,6} 或 None。"""
    if steps_per_beat is None:
        return None
    try:
        spb = int(steps_per_beat)
    except Exception:
        return None
    if spb in (1, 2, 4, 6):
        return spb
    return None


def normalize_rhythm_step_set(steps_per_beat: Optional[object]) -> Tuple[int, ...]:
    """EN: Normalize rhythm-grid selector to a validated subdivision set.
    中文：将节奏网格选择标准化为合法细分集合。"""
    if steps_per_beat is None:
        return (4, 6)

    if isinstance(steps_per_beat, (list, tuple, set)):
        out: List[int] = []
        for x in steps_per_beat:
            n = normalize_rhythm_steps_per_beat(x)
            if n is not None:
                out.append(int(n))
        if not out:
            return (4, 6)
        return tuple(sorted(set(out)))

    n = normalize_rhythm_steps_per_beat(steps_per_beat)
    if n is None:
        return (4, 6)
    return (int(n),)


def choose_global_rhythm_steps_per_beat(activity: float, syncopation: float) -> int:
    """EN: Choose one shared meter subdivision for all modules.
    中文：为全部模块选择同一细分拍感。"""
    a = _clamp01(activity)
    s = _clamp01(syncopation)
    # [EN] Higher syncopation/activity prefers 1/24 feel; otherwise keep 1/16.
    # [中文] 切分/活动较高时使用 1/24，否则保持 1/16。
    if s >= 0.58 or (s >= 0.52 and a >= 0.66):
        return 6
    return 4


def _rhythm_grid_points_ticks(ticks_per_beat: int, steps_per_beat: Optional[object] = None) -> List[int]:
    """EN: In-beat rhythm grid from selected subdivision set.
    中文：按细分集合生成单拍网格。"""
    tpb = int(max(1, ticks_per_beat))
    step_set = normalize_rhythm_step_set(steps_per_beat)
    pts = {0, tpb}
    for spb in step_set:
        spb_i = int(max(1, spb))
        for k in range(1, spb_i):
            pts.add(int(round((k * tpb) / float(spb_i))))
    return sorted(int(max(0, min(tpb, p))) for p in pts)


def minimum_rhythm_step_ticks(ticks_per_beat: int, steps_per_beat: Optional[object] = None) -> int:
    """EN: Return the minimum positive step on the active rhythm grid.
    中文：返回当前节奏网格的最小正步长。"""
    grid = _rhythm_grid_points_ticks(ticks_per_beat, steps_per_beat=steps_per_beat)
    diffs = [b - a for a, b in zip(grid[:-1], grid[1:]) if b > a]
    return int(max(1, min(diffs) if diffs else 1))


def next_rhythm_grid_tick(tick: int, ticks_per_beat: int, steps_per_beat: Optional[object] = None) -> int:
    """EN: Return the next strict grid tick after `tick`.
    中文：返回严格大于 `tick` 的下一个网格 tick。"""
    tpb = int(max(1, ticks_per_beat))
    t = int(max(0, tick))
    beat = t // tpb
    rem = t - beat * tpb
    grid = _rhythm_grid_points_ticks(tpb, steps_per_beat=steps_per_beat)
    for g in grid:
        if g > rem:
            return int(beat * tpb + g)
    # [EN] Wrap to next beat.
    # [中文] 回绕到下一拍。
    first_pos = min((g for g in grid if g > 0), default=tpb)
    return int((beat + 1) * tpb + first_pos)


def quantize_tick_to_rhythm_grid(
    tick: int,
    ticks_per_beat: int,
    steps_per_beat: Optional[object] = None,
) -> int:
    """EN: Quantize absolute tick to nearest active rhythm-grid point.
    中文：把绝对 tick 量化到当前节奏网格的最近点。"""
    tpb = int(max(1, ticks_per_beat))
    t = int(max(0, tick))
    beat = t // tpb
    rem = t - beat * tpb
    grid = _rhythm_grid_points_ticks(tpb, steps_per_beat=steps_per_beat)
    best = min(grid, key=lambda g: abs(g - rem))
    if best >= tpb:
        beat += 1
        best = 0
    return int(beat * tpb + best)


def quantize_note_window(
    start_tick: int,
    end_tick: int,
    ticks_per_beat: int,
    steps_per_beat: Optional[object] = None,
) -> Tuple[int, int]:
    """EN: Quantize note start/end and enforce a safe minimum duration.
    中文：量化音符起止时间并保证安全最小时值。"""
    tpb = int(max(1, ticks_per_beat))
    min_dur = minimum_rhythm_step_ticks(tpb, steps_per_beat=steps_per_beat)
    s = quantize_tick_to_rhythm_grid(int(start_tick), tpb, steps_per_beat=steps_per_beat)
    e = quantize_tick_to_rhythm_grid(int(end_tick), tpb, steps_per_beat=steps_per_beat)
    if e <= s:
        e = s + min_dur
    return s, e


def fit_note_to_range_preserve_pc(note: int, low: int, high: int, prefer: Optional[float] = None) -> int:
    """EN: Fit note into range by octave-shifting, preserving pitch class.
    中文：通过八度平移将音符放入目标音域，并保留音级（pitch class）。"""
    lo = int(min(low, high))
    hi = int(max(low, high))
    n = int(note)
    candidates: List[int] = []
    for k in range(-16, 17):
        m = n + 12 * k
        if lo <= m <= hi:
            candidates.append(int(m))
    if not candidates:
        return int(max(lo, min(hi, n)))
    target = float(n if prefer is None else prefer)
    return int(min(candidates, key=lambda m: (abs(m - target), abs(m - n))))


def segment_color_span(feat: SegmentFeatures) -> float:
    """EN: Estimate per-segment palette span (0..1) from color distribution.
    中文：估计每段颜色跨度（0..1），用于控制和声推进速度。"""
    sep = 0.0
    sat_sep = 0.0
    val_sep = 0.0
    if feat.secondary is not None:
        sep = circular_diff_deg(feat.primary.hue, feat.secondary.hue) / 180.0
        sat_sep = abs(float(feat.primary.sat) - float(feat.secondary.sat))
        val_sep = abs(float(feat.primary.val) - float(feat.secondary.val))
    span = (
        0.52 * _clamp01(sep)
        + 0.20 * _clamp01(feat.hue_var)
        + 0.16 * _clamp01(sat_sep + feat.sat_var * 2.0)
        + 0.12 * _clamp01(val_sep + feat.val_var * 2.0)
    )
    return _clamp01(span)


def adapt_harmonic_progression_rate(
    plans: Sequence[SegmentModulePlan],
    ticks_per_beat: int,
    rhythm_steps_per_beat: Optional[object] = None,
) -> List[SegmentModulePlan]:
    """EN: Slow down progression under stable color span; speed up under volatile span.
    中文：颜色跨度稳定时减慢和声推进，变化剧烈时加快推进。"""
    def strong_step_ticks(tpb: int) -> int:
        # [EN] Strong harmonic anchors: beat 1 and beat 3 in a 4/4 bar (every 2 beats).
        # [中文] 和声强锚点：4/4 中第 1、3 拍（每 2 拍一次）。
        return int(max(1, 2 * int(max(1, tpb))))

    def quantize_to_strong_grid(tick: int, tpb: int) -> int:
        step = strong_step_ticks(tpb)
        return int(round(int(tick) / float(step)) * step)

    src = list(plans)
    if not src:
        return []
    if len(src) == 1:
        return src

    min_step = minimum_rhythm_step_ticks(ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
    strong_step = strong_step_ticks(ticks_per_beat)
    spans = [float(p.color_span) for p in src]
    deltas = [0.0] + [abs(spans[i] - spans[i - 1]) for i in range(1, len(spans))]
    base = deltas[1:] if len(deltas) > 1 else [0.0]
    low_th = float(np.quantile(base, 0.35)) if base else 0.08
    high_th = float(np.quantile(base, 0.80)) if base else 0.35
    low_th = max(0.03, min(0.16, low_th))
    high_th = max(low_th + 0.08, min(0.55, high_th))

    out: List[SegmentModulePlan] = []
    vol_streak = 0
    calm_streak = 0
    for i, p in enumerate(src):
        d = float(deltas[i])
        if d >= high_th:
            vol_streak += 1
            calm_streak = 0
        elif d <= low_th:
            calm_streak += 1
            vol_streak = max(0, vol_streak - 1)
        else:
            vol_streak = max(0, vol_streak - 1)
            calm_streak = 0

        allow_subbeat = bool(vol_streak >= 2)
        if allow_subbeat:
            s_q, e_q = quantize_note_window(
                int(p.start_tick),
                int(p.end_tick),
                ticks_per_beat,
                steps_per_beat=rhythm_steps_per_beat,
            )
        else:
            # [EN] Default harmonic pacing is constrained to strong beats.
            # [中文] 默认和声节奏约束在强拍锚点。
            s_q = quantize_to_strong_grid(int(p.start_tick), ticks_per_beat)
            e_q = quantize_to_strong_grid(int(p.end_tick), ticks_per_beat)
            if e_q <= s_q:
                e_q = int(s_q + strong_step)
        p_q = SegmentModulePlan(
            start_tick=int(s_q),
            end_tick=int(e_q),
            harmony_notes=tuple(p.harmony_notes),
            harmony_velocity=int(p.harmony_velocity),
            bass_note=int(p.bass_note),
            bass_velocity=int(p.bass_velocity),
            pad_notes=tuple(p.pad_notes),
            pad_velocity=int(p.pad_velocity),
            color_span=float(p.color_span),
        )

        # [EN] Stable span: keep previous harmony longer (slower progression).
        # [中文] 色彩跨度稳定：延长上一和声（减慢推进）。
        if out and calm_streak >= 2:
            prev = out[-1]
            out[-1] = SegmentModulePlan(
                start_tick=int(prev.start_tick),
                end_tick=int(max(prev.end_tick, p_q.end_tick)),
                harmony_notes=tuple(prev.harmony_notes),
                harmony_velocity=int(prev.harmony_velocity),
                bass_note=int(prev.bass_note),
                bass_velocity=int(prev.bass_velocity),
                pad_notes=tuple(prev.pad_notes),
                pad_velocity=int(prev.pad_velocity),
                color_span=float(max(prev.color_span, p_q.color_span)),
            )
            continue

        dur = int(max(1, p_q.end_tick - p_q.start_tick))
        # [EN] Volatile span: split the window and add a light in-segment harmonic motion.
        # [中文] 色彩跨度剧烈：拆分该窗口并添加轻微段内和声运动。
        if allow_subbeat and d >= high_th and dur >= 2 * min_step:
            mid = quantize_tick_to_rhythm_grid(
                int(p_q.start_tick + dur // 2),
                ticks_per_beat,
                steps_per_beat=rhythm_steps_per_beat,
            )
            if mid <= p_q.start_tick:
                mid = int(p_q.start_tick + min_step)
            if mid >= p_q.end_tick:
                mid = int(p_q.end_tick - min_step)

            if p_q.start_tick < mid < p_q.end_tick:
                out.append(
                    SegmentModulePlan(
                        start_tick=int(p_q.start_tick),
                        end_tick=int(mid),
                        harmony_notes=tuple(p_q.harmony_notes),
                        harmony_velocity=int(p_q.harmony_velocity),
                        bass_note=int(p_q.bass_note),
                        bass_velocity=int(p_q.bass_velocity),
                        pad_notes=tuple(p_q.pad_notes),
                        pad_velocity=int(p_q.pad_velocity),
                        color_span=float(p_q.color_span),
                    )
                )

                harm2 = tuple(list(p_q.harmony_notes[1:]) + [p_q.harmony_notes[0]]) if len(p_q.harmony_notes) > 1 else tuple(p_q.harmony_notes)
                bass2 = int(p_q.bass_note)
                if harm2:
                    bass2 = fit_note_to_range_preserve_pc(int(harm2[0]), BASS_RANGE[0], BASS_RANGE[1], prefer=float(p_q.bass_note) + 2.0)

                pad2 = tuple(p_q.pad_notes)
                if len(p_q.pad_notes) >= 2:
                    root2 = fit_note_to_range_preserve_pc(int(harm2[0] if harm2 else p_q.pad_notes[0]), PAD_RANGE[0], PAD_RANGE[1], prefer=float(p_q.pad_notes[0]) + 2.0)
                    fifth2 = fit_note_to_range_preserve_pc(int(root2 + 7), PAD_RANGE[0], PAD_RANGE[1], prefer=float(p_q.pad_notes[1]))
                    pad_list = [root2, fifth2]
                    if len(p_q.pad_notes) >= 3:
                        pad_list.append(fit_note_to_range_preserve_pc(int(root2 + 12), PAD_RANGE[0], PAD_RANGE[1], prefer=float(p_q.pad_notes[-1])))
                    pad2 = tuple(sorted(set(pad_list)))

                out.append(
                    SegmentModulePlan(
                        start_tick=int(mid),
                        end_tick=int(p_q.end_tick),
                        harmony_notes=tuple(harm2 if harm2 else p_q.harmony_notes),
                        harmony_velocity=int(p_q.harmony_velocity),
                        bass_note=int(bass2),
                        bass_velocity=int(p_q.bass_velocity),
                        pad_notes=tuple(pad2),
                        pad_velocity=int(p_q.pad_velocity),
                        color_span=float(p_q.color_span),
                    )
                )
                continue

        out.append(p_q)

    # [EN] Cleanup: enforce monotonic windows and minimum duration.
    # [中文] 收尾：确保窗口单调不重叠，并满足最小时值。
    cleaned: List[SegmentModulePlan] = []
    last_end = 0
    for p in out:
        s = int(max(p.start_tick, last_end))
        e = int(max(s + min_step, p.end_tick))
        cleaned.append(
            SegmentModulePlan(
                start_tick=s,
                end_tick=e,
                harmony_notes=tuple(p.harmony_notes),
                harmony_velocity=int(p.harmony_velocity),
                bass_note=int(p.bass_note),
                bass_velocity=int(p.bass_velocity),
                pad_notes=tuple(p.pad_notes),
                pad_velocity=int(p.pad_velocity),
                color_span=float(p.color_span),
            )
        )
        last_end = e
    return cleaned


def build_segment_module_plans(
    segments: Sequence[SegmentInfo],
    scale: Sequence[int],
    tensions: Sequence[float],
    tension_factor: float,
    sensitivity: float,
    mood_settings: dict,
    ticks_per_frame: float,
    bars_per_segment: int,
    ticks_per_beat: int,
    rhythm_steps_per_beat: Optional[object] = None,
) -> List[SegmentModulePlan]:
    """EN: Build per-segment plans shared by Harmony/Bass/Pad modules.
    中文：构建供和声/低音/铺底模块共享的分段计划。"""
    plans: List[SegmentModulePlan] = []
    last_end = 0
    prev_root_idx: Optional[int] = None
    harmonic_energy = 0
    scale_n = int(max(1, len(scale)))
    min_step = minimum_rhythm_step_ticks(ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
    for idx, segment in enumerate(segments):
        seg_frames = max(1, segment.end - segment.start)
        segment_ticks_raw = int(max(1.0, seg_frames * ticks_per_frame * max(1, bars_per_segment)))
        start_raw = int(round(segment.start * ticks_per_frame * max(1, bars_per_segment)))
        start, end = quantize_note_window(
            start_raw,
            start_raw + segment_ticks_raw,
            ticks_per_beat,
            steps_per_beat=rhythm_steps_per_beat,
        )
        if start < last_end:
            start = int(last_end)
        if end <= start:
            end = int(start + min_step)
        last_end = end

        feat = segment.features
        hue_boost = (feat.hue_var * 60.0 + feat.sat_var * 30.0 + feat.val_var * 30.0) * max(0.5, sensitivity)
        hue_for_pitch = (feat.hue_mean + hue_boost) % 360.0
        pc_raw, octave = pitch_from_hue(hue_for_pitch)
        pc = quantize_pc(pc_raw, scale)
        root_idx_raw = scale_degree_index(pc, scale)
        motion_level = _clamp01(
            0.50 * segment_color_span(feat) +
            0.50 * _clamp01(0.6 * feat.hue_var + 1.8 * feat.sat_var + 1.8 * feat.val_var)
        )
        if motion_level >= 0.62:
            harmonic_energy = min(3, harmonic_energy + 1)
        elif motion_level <= 0.36:
            harmonic_energy = max(0, harmonic_energy - 1)

        if prev_root_idx is None:
            root_idx = int(root_idx_raw) % scale_n
        else:
            allowed = harmonic_allowed_next_degrees(int(prev_root_idx), scale_n)
            if motion_level < 0.28 and harmonic_energy == 0:
                root_idx = int(prev_root_idx)
            else:
                candidate_pool = sorted(set([int(prev_root_idx), *allowed]))
                # [EN] Require sustained visual motion before allowing larger harmonic jumps.
                # [中文] 需要持续视觉运动才允许更大和声跳转。
                if harmonic_energy >= 2:
                    candidate_pool.append(int(root_idx_raw) % scale_n)
                    candidate_pool.append(int((int(prev_root_idx) + 2) % scale_n))
                    candidate_pool.append(int((int(prev_root_idx) - 2) % scale_n))
                candidate_pool = sorted(set(candidate_pool))
                root_idx = int(
                    min(
                        candidate_pool,
                        key=lambda d: (
                            ring_degree_distance(d, int(root_idx_raw), scale_n),
                            ring_degree_distance(d, int(prev_root_idx), scale_n),
                        ),
                    )
                )
                if harmonic_energy >= 2 and ring_degree_distance(int(root_idx_raw), int(prev_root_idx), scale_n) <= 2:
                    root_idx = int(root_idx_raw) % scale_n
        root_idx = smooth_degree_progression(prev_root_idx, root_idx, scale_n, max_step=2)
        pc = int(scale[root_idx % scale_n]) % 12
        prev_root_idx = int(root_idx)
        base_vel = velocity_from_value(feat.val_mean)
        vel_boost = int((feat.sat_var + feat.val_var) * 25.0 * max(0.5, sensitivity))
        velocity = clamp_int(base_vel + vel_boost, 1, 127)
        tension = (tensions[idx] if idx < len(tensions) else 0.0) * tension_factor

        tones_pc = chord_tones(
            pc,
            scale,
            feat.sat_mean,
            tension,
            triads_only=bool(mood_settings.get("triads_only", False)),
        )
        if feat.secondary and feat.sat_mean > 0.55 and not bool(mood_settings.get("triads_only", False)):
            sec_pc, _ = pitch_from_hue(feat.secondary.hue)
            sec_pc = quantize_pc(sec_pc, scale)
            if sec_pc not in tones_pc:
                tones_pc.append(sec_pc)

        harmony_notes = []
        for t in tones_pc:
            n_raw = 12 * (octave + 1) + t
            n = fit_note_to_range_preserve_pc(n_raw, HARMONY_RANGE[0], HARMONY_RANGE[1], prefer=n_raw)
            harmony_notes.append(n)
        harmony_notes = sorted(set(harmony_notes))

        bass_raw = 12 * (max(1, octave - 2) + 1) + pc
        bass_note = fit_note_to_range_preserve_pc(bass_raw, BASS_RANGE[0], BASS_RANGE[1], prefer=bass_raw)
        bass_vel = int(max(35, velocity - 15) * float(mood_settings.get("bass_gain", 1.0)))

        pad_root_raw = 12 * (max(1, octave - 1) + 1) + pc
        pad_fifth_raw = pad_root_raw + 7
        pad_root = fit_note_to_range_preserve_pc(pad_root_raw, PAD_RANGE[0], PAD_RANGE[1], prefer=pad_root_raw)
        pad_fifth = fit_note_to_range_preserve_pc(pad_fifth_raw, PAD_RANGE[0], PAD_RANGE[1], prefer=pad_fifth_raw)
        pad_notes = [pad_root, pad_fifth]
        if mood_settings.get("add_pad_octave"):
            pad_oct = fit_note_to_range_preserve_pc(pad_root + 12, PAD_RANGE[0], PAD_RANGE[1], prefer=pad_root + 12)
            pad_notes.append(pad_oct)
        pad_notes = sorted(set(pad_notes))
        pad_vel = int(max(35, velocity - 15) * float(mood_settings.get("pad_gain", 1.0)))

        plans.append(
            SegmentModulePlan(
                start_tick=int(start),
                end_tick=int(end),
                harmony_notes=tuple(harmony_notes),
                harmony_velocity=int(velocity),
                bass_note=int(bass_note),
                bass_velocity=clamp_int(bass_vel, 1, 127),
                pad_notes=tuple(pad_notes),
                pad_velocity=clamp_int(pad_vel, 1, 127),
                color_span=float(segment_color_span(feat)),
            )
        )
    return plans


def emit_harmony_module(events: List[Tuple[int, mido.Message]], plans: Sequence[SegmentModulePlan], channel: int) -> None:
    """EN: Emit Harmony notes from shared segment plans.
    中文：根据分段计划输出和声轨。"""
    for p in plans:
        dur = max(1, int(p.end_tick - p.start_tick))
        for note in p.harmony_notes:
            schedule_note(events, int(p.start_tick), dur, int(note), clamp_int(int(p.harmony_velocity), 1, 127), channel=channel)


def emit_bass_module(events: List[Tuple[int, mido.Message]], plans: Sequence[SegmentModulePlan], channel: int) -> None:
    """EN: Emit Bass notes from shared segment plans.
    中文：根据分段计划输出低音轨。"""
    for p in plans:
        dur = max(1, int(p.end_tick - p.start_tick))
        schedule_note(events, int(p.start_tick), dur, int(p.bass_note), clamp_int(int(p.bass_velocity), 1, 127), channel=channel)


def emit_pad_module(events: List[Tuple[int, mido.Message]], plans: Sequence[SegmentModulePlan], channel: int) -> None:
    """EN: Emit Pad notes from shared segment plans.
    中文：根据分段计划输出铺底轨。"""
    for p in plans:
        dur = max(1, int(p.end_tick - p.start_tick))
        top_note = max(p.pad_notes) if len(p.pad_notes) >= 3 else None
        for note in p.pad_notes:
            vel = int(max(1, p.pad_velocity - 10)) if top_note is not None and note == top_note else int(p.pad_velocity)
            schedule_note(events, int(p.start_tick), dur, int(note), clamp_int(vel, 1, 127), channel=channel)


def quantize_track_note_events(
    events: Sequence[Tuple[int, mido.Message]],
    ticks_per_beat: int,
    channel: int,
    note_low: int,
    note_high: int,
    rhythm_steps_per_beat: Optional[object] = None,
) -> List[Tuple[int, mido.Message]]:
    """EN: Post-quantize one track's note events onto the rhythm grid.
    中文：将单轨音符事件在后处理中量化到节奏网格。"""
    out: List[Tuple[int, mido.Message]] = []
    for t, msg in events:
        if msg.type not in ("note_on", "note_off"):
            out.append((int(t), msg.copy(time=0)))
    for start, end, note, vel in collect_note_intervals(events):
        qs, qe = quantize_note_window(start, end, ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
        qn = fit_note_to_range_preserve_pc(note, note_low, note_high, prefer=note)
        schedule_note(out, int(qs), max(1, int(qe - qs)), int(qn), clamp_int(int(vel), 1, 127), channel=channel)
    out.sort(key=lambda e: e[0])
    return out


def _active_pitch_classes_at_tick(intervals: Sequence[Tuple[int, int, int, int]], tick: int) -> List[int]:
    """EN: Gather active pitch classes at a given absolute tick.
    中文：获取某个绝对 tick 时刻的活动音级集合。"""
    t = int(tick)
    return [int(note) % 12 for s, e, note, _ in intervals if int(s) <= t < int(e)]


def _is_dissonant_pc(note_pc: int, active_pcs: Sequence[int]) -> bool:
    """EN: Heuristic dissonance test against concurrently active notes.
    中文：与并发音比较的启发式不协和判定。"""
    for p in active_pcs:
        d = min((int(note_pc) - int(p)) % 12, (int(p) - int(note_pc)) % 12)
        if d in (1, 2, 6):
            return True
    return False


def fit_note_to_allowed_pcs(
    note: int,
    allowed_pcs: Sequence[int],
    low: int,
    high: int,
    prefer: Optional[float] = None,
) -> int:
    """EN: Fit note into range while constraining to a set of allowed pitch classes.
    中文：在给定音域内将音符约束到指定音级集合。"""
    pcs = sorted(set(int(p) % 12 for p in allowed_pcs))
    if not pcs:
        return fit_note_to_range_preserve_pc(int(note), int(low), int(high), prefer=prefer)
    lo = int(min(low, high))
    hi = int(max(low, high))
    target = float(note if prefer is None else prefer)
    candidates: List[int] = [n for n in range(lo, hi + 1) if (n % 12) in pcs]
    if not candidates:
        return fit_note_to_range_preserve_pc(int(note), lo, hi, prefer=target)
    return int(min(candidates, key=lambda n: (abs(n - target), abs(n - int(note)))))


def resolve_melody_clashes_after_quantization(
    melody_events: Sequence[Tuple[int, mido.Message]],
    harmony_events: Sequence[Tuple[int, mido.Message]],
    bass_events: Sequence[Tuple[int, mido.Message]],
    pad_events: Sequence[Tuple[int, mido.Message]],
    *,
    ticks_per_beat: int,
    melody_channel: int,
    total_ticks: int,
    scale_pcs_at_tick_fn,
    rhythm_steps_per_beat: Optional[object] = None,
) -> List[Tuple[int, mido.Message]]:
    """EN: Two-stage melody repair: timing-first, then pitch-if-needed.
    中文：两阶段旋律修正：先时间，后音高（必要时）。"""
    out: List[Tuple[int, mido.Message]] = []
    for t, msg in melody_events:
        if msg.type not in ("note_on", "note_off"):
            out.append((int(t), msg.copy(time=0)))

    harmony_intervals = collect_note_intervals(list(harmony_events) + list(bass_events) + list(pad_events))
    melody_intervals = collect_note_intervals(melody_events)
    min_step = minimum_rhythm_step_ticks(ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
    max_delay = min_step * 6
    tpb = int(max(1, ticks_per_beat))

    def is_strong_harmonic_tick(tick: int) -> bool:
        t = int(max(0, tick))
        if (t % tpb) != 0:
            return False
        beat_in_bar = (t // tpb) % 4
        return beat_in_bar in (0, 2)

    for s, e, note, vel in melody_intervals:
        # [EN] Stage A: timing repair (on-grid delay) without touching pitch.
        # [中文] 阶段A：仅做时间修正（网格后移），不改音高。
        start, end = quantize_note_window(s, e, ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
        dur = max(min_step, int(end - start))
        base = int(start)
        delayed = False
        while True:
            if start >= int(total_ticks):
                break
            active_pcs = _active_pitch_classes_at_tick(harmony_intervals, start)
            local_scale_pcs = scale_pcs_at_tick_fn(int(start))
            note_pc = int(note) % 12
            out_of_scale = note_pc not in local_scale_pcs
            clash = _is_dissonant_pc(note_pc, active_pcs)
            if (not out_of_scale) and (not clash):
                break
            nxt = next_rhythm_grid_tick(start, ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
            if nxt - base > max_delay:
                break
            start = nxt
            delayed = True

        if start >= int(total_ticks):
            continue
        if delayed:
            # [EN] Reclassified as a stepping note: keep it short.
            # [中文] 重新归类为经过音：时值缩短。
            dur = min(dur, min_step)
        end2 = int(min(int(total_ticks), start + max(min_step, dur)))
        if end2 <= start:
            end2 = int(start + min_step)

        # [EN] Stage B: pitch repair only if timing repair still leaves clash/out-of-scale.
        # [中文] 阶段B：仅当时间修正后仍冲突/离调时才改音高。
        active_pcs_final = _active_pitch_classes_at_tick(harmony_intervals, start)
        local_scale_pcs_final = set(scale_pcs_at_tick_fn(int(start)))
        note_out = fit_note_to_range_preserve_pc(int(note), MELODY_RANGE[0], MELODY_RANGE[1], prefer=note)
        note_out_pc = int(note_out) % 12
        out_of_scale_final = note_out_pc not in local_scale_pcs_final if local_scale_pcs_final else False
        clash_final = _is_dissonant_pc(note_out_pc, active_pcs_final)
        if out_of_scale_final or clash_final:
            strong_tick = is_strong_harmonic_tick(start)
            scale_pool = sorted(local_scale_pcs_final) if local_scale_pcs_final else []
            harmonic_pool = [pc for pc in sorted(set(active_pcs_final)) if (not local_scale_pcs_final) or (pc in local_scale_pcs_final)]
            chosen_pool: List[int] = []
            if strong_tick:
                # [EN] Strong beats prefer chord tones.
                # [中文] 强拍优先使用和弦音。
                chosen_pool = harmonic_pool if harmonic_pool else scale_pool
            else:
                # [EN] Weak beats prefer scale passing tones while avoiding severe clashes.
                # [中文] 弱拍优先调内经过音，并尽量回避强冲突。
                chosen_pool = scale_pool if scale_pool else harmonic_pool
                if active_pcs_final and chosen_pool:
                    safe = [pc for pc in chosen_pool if not _is_dissonant_pc(pc, active_pcs_final)]
                    if safe:
                        chosen_pool = safe
            if chosen_pool:
                note_out = fit_note_to_allowed_pcs(
                    note_out,
                    chosen_pool,
                    MELODY_RANGE[0],
                    MELODY_RANGE[1],
                    prefer=float(note_out),
                )
        schedule_note(
            out,
            int(start),
            int(max(1, end2 - start)),
            int(note_out),
            clamp_int(int(vel), 1, 127),
            channel=melody_channel,
        )
    out.sort(key=lambda e: e[0])
    return out


def _ioi_entropy_norm(onsets: Sequence[int]) -> float:
    """EN: Normalized entropy of onset intervals (0=very repetitive, 1=very random).
    中文：起音间隔归一化熵（0=高度重复，1=高度随机）。"""
    xs = sorted(set(int(x) for x in onsets))
    if len(xs) < 3:
        return 0.0
    ioi = np.diff(np.asarray(xs, dtype=np.int64))
    ioi = ioi[ioi > 0]
    if ioi.size <= 1:
        return 0.0
    vals, cnt = np.unique(ioi, return_counts=True)
    p = cnt.astype(np.float64) / float(np.sum(cnt))
    ent = float(-np.sum(p * np.log2(np.maximum(p, 1e-12))))
    max_ent = math.log2(max(2, len(vals)))
    return float(max(0.0, min(1.0, ent / max_ent)))


def evaluate_post_metrics(
    harmony_events: Sequence[Tuple[int, mido.Message]],
    bass_events: Sequence[Tuple[int, mido.Message]],
    pad_events: Sequence[Tuple[int, mido.Message]],
    melody_events: Sequence[Tuple[int, mido.Message]],
) -> dict:
    """EN: Compute post-production quality metrics for fallback decisions.
    中文：计算后处理质量指标，用于回退决策。"""
    accompaniment = collect_note_intervals(list(harmony_events) + list(bass_events) + list(pad_events))
    melody = collect_note_intervals(melody_events)
    if not melody:
        return {
            "dissonance_ratio": 0.0,
            "collision_ratio": 0.0,
            "ioi_entropy_norm": 0.0,
            "repetition_ratio": 1.0,
        }
    dis = 0
    coll = 0
    onset_set = {int(s) for s, _, _, _ in accompaniment}
    onsets = []
    for s, _, note, _ in melody:
        ss = int(s)
        onsets.append(ss)
        active = _active_pitch_classes_at_tick(accompaniment, ss)
        if active and _is_dissonant_pc(int(note) % 12, active):
            dis += 1
        if ss in onset_set:
            coll += 1
    n = float(max(1, len(melody)))
    ent = _ioi_entropy_norm(onsets)
    return {
        "dissonance_ratio": float(dis / n),
        "collision_ratio": float(coll / n),
        "ioi_entropy_norm": float(ent),
        "repetition_ratio": float(1.0 - ent),
    }


def safe_fallback_thresholds(mapping_bias: float) -> dict:
    """EN: Convert one safety-vs-visual scalar to post-fallback trigger thresholds.
    中文：将单一“安全-视觉”标量映射为后处理回退触发阈值。"""
    b = _clamp01(float(mapping_bias))
    # [EN] Keep legacy behavior at b=0.5 while allowing stricter/looser fallback at extremes.
    # [中文] 在 b=0.5 时保持旧版本默认行为，并在两端提供更严格/更宽松回退。
    return {
        "dissonance_ratio_max": float(0.16 + 0.12 * b),  # b=0.5 -> 0.22
        "collision_ratio_max": float(0.35 + 0.20 * b),   # b=0.5 -> 0.45
        "ioi_entropy_max": float(0.86 + 0.12 * b),       # b=0.5 -> 0.92
        "ioi_entropy_min": float(0.14 - 0.12 * b),       # b=0.5 -> 0.08
    }


def maybe_apply_safe_post_fallback(
    *,
    harmony_events: List[Tuple[int, mido.Message]],
    bass_events: List[Tuple[int, mido.Message]],
    pad_events: List[Tuple[int, mido.Message]],
    melody_events: List[Tuple[int, mido.Message]],
    ticks_per_beat: int,
    ch_harmony: int,
    ch_bass: int,
    ch_pad: int,
    ch_melody: int,
    total_ticks: int,
    scale_pcs_at_tick_fn,
    mapping_bias: float = 0.5,
) -> Tuple[List[Tuple[int, mido.Message]], List[Tuple[int, mido.Message]], List[Tuple[int, mido.Message]], List[Tuple[int, mido.Message]], dict, bool]:
    """EN: Auto-fallback to safer rhythmic settings when post metrics are poor.
    中文：当后处理指标不佳时自动回退到更保守的节奏设置。"""
    m0 = evaluate_post_metrics(harmony_events, bass_events, pad_events, melody_events)
    th = safe_fallback_thresholds(mapping_bias)
    trigger = (
        float(m0["dissonance_ratio"]) > float(th["dissonance_ratio_max"])
        or float(m0["collision_ratio"]) > float(th["collision_ratio_max"])
        or float(m0["ioi_entropy_norm"]) > float(th["ioi_entropy_max"])
        or float(m0["ioi_entropy_norm"]) < float(th["ioi_entropy_min"])
    )
    if not trigger:
        return harmony_events, bass_events, pad_events, melody_events, m0, False

    h2 = quantize_track_note_events(harmony_events, ticks_per_beat, ch_harmony, HARMONY_RANGE[0], HARMONY_RANGE[1], rhythm_steps_per_beat=1)
    b2 = quantize_track_note_events(bass_events, ticks_per_beat, ch_bass, BASS_RANGE[0], BASS_RANGE[1], rhythm_steps_per_beat=1)
    p2 = quantize_track_note_events(pad_events, ticks_per_beat, ch_pad, PAD_RANGE[0], PAD_RANGE[1], rhythm_steps_per_beat=1)
    m2 = quantize_track_note_events(melody_events, ticks_per_beat, ch_melody, MELODY_RANGE[0], MELODY_RANGE[1], rhythm_steps_per_beat=4)
    m2 = resolve_melody_clashes_after_quantization(
        m2,
        h2,
        b2,
        p2,
        ticks_per_beat=ticks_per_beat,
        melody_channel=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick_fn,
        rhythm_steps_per_beat=4,
    )
    m2 = quantize_track_note_events(m2, ticks_per_beat, ch_melody, MELODY_RANGE[0], MELODY_RANGE[1], rhythm_steps_per_beat=4)

    m1 = evaluate_post_metrics(h2, b2, p2, m2)
    risk0 = float(m0["dissonance_ratio"]) + float(m0["collision_ratio"])
    risk1 = float(m1["dissonance_ratio"]) + float(m1["collision_ratio"])
    if risk1 <= risk0:
        return h2, b2, p2, m2, m1, True
    return harmony_events, bass_events, pad_events, melody_events, m0, False


def emit_melody_module(
    events: List[Tuple[int, mido.Message]],
    *,
    segments: Sequence[SegmentInfo],
    accent: Sequence[float],
    harmony_events: Sequence[Tuple[int, mido.Message]],
    bass_events: Sequence[Tuple[int, mido.Message]],
    pad_events: Sequence[Tuple[int, mido.Message]],
    total_frames: int,
    ticks_per_frame: float,
    total_ticks: int,
    ticks_per_beat: int,
    phrase_beats: float,
    seed: int,
    mu: float,
    sigma: float,
    sensitivity: float,
    mood_settings: dict,
    scale: Sequence[int],
    channel: int,
    rhythm_steps_per_beat: Optional[object] = None,
) -> None:
    """EN: Emit melody from internally-derived phrase templates (no external banks).
    中文：使用内部推导的乐句模板生成旋律（不依赖外部模板库）。"""
    if total_frames <= 0:
        total_frames = 1
    step_set = normalize_rhythm_step_set(rhythm_steps_per_beat)
    melody_spb = int(max(step_set))
    melody_step_ticks = minimum_rhythm_step_ticks(ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
    phrase_ticks = int(round(float(phrase_beats) * ticks_per_beat))
    phrase_steps = int(max(1, round(float(phrase_beats) * melody_spb)))
    bar_steps = int(max(1, round(4.0 * melody_spb)))
    bars_per_phrase = int(max(1, round(float(phrase_beats) / 4.0)))
    rng = np.random.default_rng(seed)
    global_valence, global_arousal = global_affect(segments)
    global_activity = _clamp01(mu + sigma * 2.5)
    base_density, base_sync, base_beta = rhythm_targets_from_affect(global_valence, global_arousal, global_activity)
    split_prob_global = split_probability(mu, sigma)

    prev_hue = None
    prev_f_param = None
    prev_melody_note = None
    step_counter = 0

    seg_idx = 0
    segments_sorted = list(segments)
    segments_sorted.sort(key=lambda s: s.start)

    def features_for_frame(frame_idx: int) -> SegmentFeatures:
        nonlocal seg_idx
        if not segments_sorted:
            return segments[0].features
        while seg_idx + 1 < len(segments_sorted) and frame_idx >= segments_sorted[seg_idx].end:
            seg_idx += 1
        return segments_sorted[seg_idx].features

    accompaniment_intervals = collect_note_intervals(list(harmony_events) + list(bass_events) + list(pad_events))
    accompaniment_onsets = {int(s) for s, _, _, _ in accompaniment_intervals}

    def onset_steps_from_events(events_beat: Sequence[Tuple[float, float]], beats: float) -> List[int]:
        total = int(max(1, round(float(beats) * melody_spb)))
        out: List[int] = []
        for st, _ in events_beat:
            step = int(round(float(st) * melody_spb))
            if 0 <= step < total:
                out.append(step)
        return sorted(set(out))

    def prune_steps_by_weight(candidates: Sequence[int], anchors: Sequence[int], target_count: int, weights: dict[int, float], total_steps_local: int) -> List[int]:
        cand = sorted(set(int(c) for c in candidates if 0 <= int(c) < total_steps_local))
        keep = set(int(a) for a in anchors if 0 <= int(a) < total_steps_local)
        ranked = sorted(cand, key=lambda x: float(weights.get(int(x), 0.0)), reverse=True)
        for idx in ranked:
            if len(keep) >= int(max(1, target_count)):
                break
            keep.add(int(idx))
        if not keep and cand:
            keep.add(int(cand[0]))
        return sorted(keep)

    def build_phrase_candidates(
        *,
        density: float,
        sync: float,
        beta: float,
        split_prob_local: float,
    ) -> List[List[int]]:
        total_steps_local = phrase_steps
        anchors_bar = [0, int(min(bar_steps - 1, 2 * melody_spb))]

        old_phrase = generate_phrase_events(split_prob_local, rng, max_depth=2)
        old_bar = [(s, d) for s, d in old_phrase if 0.0 <= float(s) < 4.0]
        old_steps = onset_steps_from_events(old_bar, 4.0)

        aff_bar = generate_phrase_events_affective(
            rng,
            density=float(density),
            syncopation=float(sync),
            beta=float(beta),
            phrase_beats=4.0,
            steps_per_beat=melody_spb,
        )
        aff_steps = onset_steps_from_events(aff_bar, 4.0)

        counts = np.zeros(bar_steps, dtype=np.float32)
        sim_n = int(max(4, round(5 + 4 * _clamp01(density + 0.5 * sync))))
        for _ in range(sim_n):
            sim = generate_phrase_events_affective(
                rng,
                density=float(density),
                syncopation=float(sync),
                beta=float(beta),
                phrase_beats=4.0,
                steps_per_beat=melody_spb,
            )
            for st in onset_steps_from_events(sim, 4.0):
                if 0 <= st < bar_steps:
                    counts[st] += 1.0

        weights: dict[int, float] = {}
        for i in range(bar_steps):
            w = float(counts[i])
            if i in old_steps:
                w += 1.3
            if i in aff_steps:
                w += 0.9
            if i in anchors_bar:
                w += 2.0
            if (i % melody_spb) == 0:
                w += 0.35
            elif (i % melody_spb) == (melody_spb // 2):
                w += 0.15
            weights[i] = float(w)

        target_bar_onsets = int(max(2, min(bar_steps, round(2.0 + 4.5 * density + 0.8 * sync))))
        first_bar_raw = sorted(set(old_steps).union(set(aff_steps)).union(set(anchors_bar)))
        first_bar = prune_steps_by_weight(first_bar_raw, anchors_bar, target_bar_onsets, weights, bar_steps)

        probable_bar_raw = sorted(range(bar_steps), key=lambda k: float(counts[k]), reverse=True)[: max(target_bar_onsets, 2)]
        probable_bar = prune_steps_by_weight(probable_bar_raw, anchors_bar, target_bar_onsets, weights, bar_steps)
        hybrid_bar = prune_steps_by_weight(
            sorted(set(first_bar).union(set(probable_bar))),
            anchors_bar,
            target_bar_onsets + 1,
            weights,
            bar_steps,
        )

        offbeat_extras = [x for x in probable_bar if (x % melody_spb) != 0]

        def expand_bar_pattern(base_bar: Sequence[int], variant: int) -> List[int]:
            out: List[int] = []
            base = sorted(set(int(x) for x in base_bar if 0 <= int(x) < bar_steps))
            if not base:
                base = [0]
            for b in range(bars_per_phrase):
                cur = list(base)
                if variant == 1 and b % 2 == 1 and sync > 0.55 and offbeat_extras:
                    cur.append(int(offbeat_extras[(b - 1) % len(offbeat_extras)]))
                if variant == 2 and b % 2 == 1:
                    cur = [int(min(bar_steps - 1, max(0, x + (1 if (x % melody_spb) == 0 else 0)))) for x in cur]
                shift = b * bar_steps
                for st in sorted(set(cur)):
                    pos = int(shift + st)
                    if 0 <= pos < total_steps_local:
                        out.append(pos)
            out = sorted(set(out))
            if 0 not in out:
                out.insert(0, 0)
            if total_steps_local >= bar_steps * 2:
                out = sorted(set(out).union({int(total_steps_local // 2)}))
            return out

        candidates: List[List[int]] = []
        candidates.append(expand_bar_pattern(first_bar, variant=0))
        candidates.append(expand_bar_pattern(probable_bar, variant=1))
        candidates.append(expand_bar_pattern(hybrid_bar, variant=2))

        aff_phrase = generate_phrase_events_affective(
            rng,
            density=float(density),
            syncopation=float(sync),
            beta=float(beta),
            phrase_beats=phrase_beats,
            steps_per_beat=melody_spb,
        )
        aff_phrase_steps = onset_steps_from_events(aff_phrase, phrase_beats)
        if aff_phrase_steps:
            candidates.append(sorted(set(aff_phrase_steps)))

        uniq: List[List[int]] = []
        seen = set()
        for c in candidates:
            key = tuple(int(x) for x in sorted(set(c)))
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(list(key))
        return uniq if uniq else [[0]]

    def score_phrase_candidate(
        onsets: Sequence[int],
        phrase_start_tick: int,
        density: float,
        prev_onsets: Optional[Sequence[int]],
    ) -> float:
        if not onsets:
            return -1e9
        ons = sorted(set(int(x) for x in onsets if 0 <= int(x) < phrase_steps))
        if not ons:
            return -1e9
        align = 0.0
        collisions = 0
        for st in ons:
            tick = int(phrase_start_tick + round((float(st) / float(max(1, melody_spb))) * ticks_per_beat))
            frame_idx = int(round(tick / max(1e-6, ticks_per_frame)))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            align += float(accent[frame_idx]) if accent and frame_idx < len(accent) else 0.0
            if int(tick) in accompaniment_onsets and ((st % melody_spb) != 0):
                collisions += 1
        align /= float(max(1, len(ons)))
        collision_pen = float(collisions) / float(max(1, len(ons)))
        expected = int(max(2, round(float(phrase_beats) * (0.45 + 0.85 * density))))
        density_pen = abs(len(ons) - expected) / float(max(1, expected))
        continuity = 0.5
        if prev_onsets:
            prev = set(int(x) for x in prev_onsets)
            cur = set(int(x) for x in ons)
            continuity = len(prev.intersection(cur)) / float(max(1, len(prev.union(cur))))
        return float(0.55 * align + 0.25 * continuity - 0.30 * collision_pen - 0.20 * density_pen)

    repeats = int(math.ceil(total_ticks / max(1, phrase_ticks)))
    template_repeat = 2 if global_activity < 0.45 else 1
    cached_template_steps: Optional[List[int]] = None
    prev_template_steps: Optional[List[int]] = None

    for r in range(repeats):
        phrase_start_ticks = r * phrase_ticks
        if phrase_start_ticks >= total_ticks:
            break
        phrase_frame_idx = int(round(phrase_start_ticks / max(1e-6, ticks_per_frame)))
        phrase_frame_idx = max(0, min(phrase_frame_idx, total_frames - 1))
        phrase_feat = features_for_frame(phrase_frame_idx)
        phrase_accent = float(accent[phrase_frame_idx]) if accent and phrase_frame_idx < len(accent) else 0.0
        phrase_activity = _clamp01(
            0.50 * phrase_accent + 0.50 * _clamp01(0.6 * phrase_feat.hue_var + 1.8 * phrase_feat.sat_var + 1.8 * phrase_feat.val_var)
        )
        phrase_valence, phrase_arousal = affect_from_segment(phrase_feat)
        loc_density, loc_sync, loc_beta = rhythm_targets_from_affect(phrase_valence, phrase_arousal, phrase_activity)
        density = _clamp01(0.65 * base_density + 0.35 * loc_density)
        sync = _clamp01(0.65 * base_sync + 0.35 * loc_sync)
        beta = max(0.35, min(1.80, 0.65 * base_beta + 0.35 * loc_beta))
        split_prob_local = float(max(0.08, min(0.75, 0.55 * split_prob_global + 0.45 * (0.12 + 0.62 * phrase_activity))))

        if (r % template_repeat) == 0 or cached_template_steps is None:
            candidates = build_phrase_candidates(
                density=density,
                sync=sync,
                beta=beta,
                split_prob_local=split_prob_local,
            )
            best = max(
                candidates,
                key=lambda cand: score_phrase_candidate(cand, phrase_start_ticks, density, prev_template_steps),
            )
            cached_template_steps = list(best)
        phrase_template_steps = list(cached_template_steps) if cached_template_steps else [0]
        prev_template_steps = list(phrase_template_steps)

        used_onsets: set[int] = set()
        for i, start_step in enumerate(phrase_template_steps):
            next_step = phrase_template_steps[i + 1] if i + 1 < len(phrase_template_steps) else phrase_steps
            raw_steps = max(1, int(next_step - start_step))
            dur_steps = raw_steps if raw_steps <= 1 else (raw_steps - 1)
            start_beat = float(start_step) / float(max(1, melody_spb))
            dur_beat = max(1.0 / float(max(1, melody_spb)), float(dur_steps) / float(max(1, melody_spb)))
            start_raw = phrase_start_ticks + int(round(start_beat * ticks_per_beat))
            end_raw = phrase_start_ticks + int(round((start_beat + dur_beat) * ticks_per_beat))
            start_ticks, end_ticks = quantize_note_window(
                start_raw,
                end_raw,
                ticks_per_beat,
                steps_per_beat=rhythm_steps_per_beat,
            )
            micro_prob = float(min(0.32, 0.06 + 0.24 * phrase_activity))
            if rng.random() < micro_prob:
                shift_dir = 1 if (phrase_feat.hue_var + phrase_feat.sat_var) >= (phrase_feat.val_var + 0.5 * phrase_feat.hue_var) else -1
                shifted = quantize_tick_to_rhythm_grid(
                    int(start_ticks + shift_dir * melody_step_ticks),
                    ticks_per_beat,
                    steps_per_beat=rhythm_steps_per_beat,
                )
                phrase_end_limit = int(min(total_ticks, phrase_start_ticks + phrase_ticks))
                if int(phrase_start_ticks) <= shifted < phrase_end_limit:
                    start_ticks = int(shifted)
                    if end_ticks <= start_ticks:
                        end_ticks = int(start_ticks + melody_step_ticks)
            if start_ticks >= total_ticks:
                continue
            while start_ticks in used_onsets:
                start_ticks = next_rhythm_grid_tick(start_ticks, ticks_per_beat, steps_per_beat=rhythm_steps_per_beat)
                if start_ticks >= total_ticks:
                    break
            if start_ticks >= total_ticks:
                continue
            used_onsets.add(int(start_ticks))
            duration_ticks = max(1, end_ticks - start_ticks)
            if start_ticks + duration_ticks > total_ticks:
                duration_ticks = max(1, total_ticks - start_ticks)

            frame_idx = int(round(start_ticks / max(1e-6, ticks_per_frame)))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            accent_level = float(accent[frame_idx]) if accent and frame_idx < len(accent) else 0.0

            feat = features_for_frame(frame_idx)
            hue_boost = (feat.hue_var * 60.0 + feat.sat_var * 30.0 + feat.val_var * 30.0) * max(0.5, sensitivity)
            hue_for_pitch = (feat.hue_mean + hue_boost) % 360.0
            pc_raw, octave = pitch_from_hue(hue_for_pitch)
            pc = quantize_pc(pc_raw, scale)
            scale_midi = scale_notes(pc, scale, octave)
            if not scale_midi:
                continue

            direction = 0
            delta_hue = 0.0
            if prev_hue is not None:
                delta_hue = ((feat.hue_mean - prev_hue + 540.0) % 360.0) - 180.0
                threshold = 10.0 / max(0.5, sensitivity)
                if delta_hue > threshold:
                    direction = 1
                elif delta_hue < -threshold:
                    direction = -1
                elif prev_f_param is not None:
                    direction = 1 if feat.f_param_mean > prev_f_param else -1
            step_span = max(1, int(round((abs(delta_hue) / 360.0) * 6 * max(0.5, sensitivity))))
            step_span = min(step_span, 2)
            micro_offset = step_span if direction >= 0 else -step_span
            prev_hue = feat.hue_mean
            prev_f_param = feat.f_param_mean

            rhythmic_hint = int(round(float(start_step) / float(max(1, melody_spb // 2 or 1))))
            note_idx = (step_counter * 2 + rhythmic_hint + direction + micro_offset) % len(scale_midi)
            max_jump = int(mood_settings.get("max_jump", 5))
            note = limit_melody_jump(scale_midi, note_idx, prev_melody_note, max_semitone_step=max_jump)
            note = fit_note_to_range_preserve_pc(note, MELODY_RANGE[0], MELODY_RANGE[1], prefer=note)
            prev_melody_note = note
            step_counter += 1

            base_vel = velocity_from_value(feat.val_mean)
            vel_boost = int((feat.sat_var + feat.val_var) * 25.0 * max(0.5, sensitivity))
            velocity = clamp_int(base_vel + vel_boost, 1, 127)
            if accent_level >= 0.75:
                velocity += 18
            elif accent_level >= 0.5:
                velocity += 8

            schedule_note(
                events,
                int(start_ticks),
                int(duration_ticks),
                int(note),
                clamp_int(velocity, 1, 127),
                channel=channel,
            )


def generate_phrase_events_affective(
    rng: np.random.Generator,
    density: float,
    syncopation: float,
    beta: float,
    *,
    phrase_beats: float = 8.0,
    steps_per_beat: Optional[object] = None,
) -> List[Tuple[float, float]]:
    """EN: Build one phrase rhythm on a strict conventional grid.
    中文：在严格常规网格上生成乐句节奏。

    EN:
    - subdivision is either straight semiquavers (1/4 beat) or 1/24-note grid (1/6 beat)
    - never generates finer placements than the selected grid
    中文：
    - 细分仅使用十六分（1/4拍）或二十四分（1/6拍）
    - 不会生成比该网格更细的位置
    """
    # [EN] If a global meter is fixed, follow it; otherwise phrase-level adaptive choice.
    # [中文] 若已固定全局拍感则直接使用；否则按乐句自适应选择。
    spb_fixed = normalize_rhythm_steps_per_beat(steps_per_beat)
    if spb_fixed is not None:
        spb = int(spb_fixed)
    else:
        # [EN] Per-phrase mode: straight 1/16 or finer 1/24 feel.
        # [中文] 每条乐句在“十六分”与“二十四分”中二选一。
        use_24 = bool(syncopation >= 0.60 and rng.random() < (0.25 + 0.50 * _clamp01(syncopation)))
        spb = 6 if use_24 else 4
    total_steps = int(max(1, round(float(phrase_beats) * spb)))
    d = _clamp01(density)
    s = _clamp01(syncopation)
    noise = _one_over_f_noise(total_steps, beta, rng)

    anchors = {0}
    if total_steps >= spb * 4:
        anchors.add(total_steps // 2)

    probs: List[float] = []
    picks: List[int] = []
    for i in range(total_steps):
        is_quarter = (i % spb) == 0
        # [EN] Mid-beat support differs between 1/16 and 1/24 grids.
        # [中文] 中拍支撑根据十六分/二十四分网格分别处理。
        is_mid = (i % spb) == (spb // 2)
        if is_quarter:
            meter_w = 1.0
        elif is_mid:
            meter_w = 0.60
        else:
            meter_w = 0.42
        sync_shift = (-0.22 * s) if is_quarter else (0.26 * s)
        p = (0.10 + 0.52 * d) * meter_w + sync_shift + 0.12 * (float(noise[i]) - 0.5)
        if i in anchors:
            p = max(p, 0.86)
        p = float(max(0.02, min(0.95, p)))
        probs.append(p)
        if i in anchors or rng.random() < p:
            picks.append(i)

    # [EN] Cap phrase density to avoid machine-gun pacing.
    # [中文] 限制单乐句音符密度，避免“机枪式”节奏。
    max_onsets = int(max(4, min(total_steps, round(5 + d * 8))))
    if len(picks) > max_onsets:
        ranked = sorted(picks, key=lambda idx: probs[idx], reverse=True)
        keep = set(anchors)
        for idx in ranked:
            if len(keep) >= max_onsets:
                break
            keep.add(idx)
        picks = sorted(keep)
    else:
        picks = sorted(set(picks).union(anchors))

    if not picks:
        picks = [0]

    # [EN] Enforce minimum one-grid-step spacing and strict quantization.
    # [中文] 强制最小一网格步长间距并保持严格量化。
    cleaned: List[int] = []
    last = -999999
    for step in sorted(set(picks)):
        if step - last >= 1:
            cleaned.append(step)
            last = step
    picks = cleaned if cleaned else [0]

    events: List[Tuple[float, float]] = []
    for i, start_step in enumerate(picks):
        next_step = picks[i + 1] if i + 1 < len(picks) else total_steps
        raw_steps = max(1, next_step - start_step)
        # [EN] Keep duration on the same grid; leave a one-step breath when possible.
        # [中文] 时值保持在同一网格；在可能时留出一步呼吸。
        dur_steps = raw_steps if raw_steps <= 1 else (raw_steps - 1)
        start_beat = start_step / float(spb)
        dur_beat = max(1.0 / float(spb), dur_steps / float(spb))
        events.append((start_beat, dur_beat))
    return events


def circular_diff_deg(a: float, b: float) -> float:
    """EN: Shortest wrapped angular distance in degrees.
    中文：角度环形最短距离（度）。

    EN: Returns absolute angular difference on a circle, always in [0,180].
    中文：返回圆周意义下的绝对角差，范围恒为 [0,180]。
    """
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def smooth_series(values: Sequence[float], window: int) -> List[float]:
    """EN: Moving-average smoothing with odd window and edge padding.
    中文：奇数窗口滑动平均，边界采用边缘值填充。

    EN:
    - force odd window to keep center alignment
    - pad both ends with edge values to preserve sequence length
    中文：
    - 强制奇数窗口以保持中心对齐
    - 两端用边缘值填充，保证输出长度与输入一致
    """
    if not values:
        return []
    window = max(1, int(window))
    if window <= 1:
        return [float(v) for v in values]
    if window % 2 == 0:
        window += 1
    half = window // 2
    # [EN] Edge padding avoids shrinking and reduces boundary bias.
    # [中文] 边缘填充可避免长度收缩，并减小边界偏差。
    padded = [values[0]] * half + list(values) + [values[-1]] * half
    out = []
    for i in range(half, len(padded) - half):
        out.append(float(np.mean(padded[i - half : i + half + 1])))
    return out


def build_signal_s(frames: Sequence[np.ndarray], smooth_window: int) -> Tuple[List[float], List[float], List[float]]:
    """Purpose / 目的
    EN: Build the low-frequency visual activity signal S(t), the core rhythm driver.
    中文：构建低频视觉活动信号 S(t)，作为节奏生成核心驱动。

    Inputs / 输入
    EN: HSV frame sequence and smoothing window size.
    中文：HSV 帧序列与平滑窗口大小。

    Method / 方法
    EN:
    1) compute per-frame mean brightness and circular-mean hue
    2) compute frame-to-frame deltas
    3) normalize brightness/hue deltas separately
    4) fuse with weighted sum
    5) smooth to suppress jitter
    中文：
    1）逐帧计算亮度均值与色相环形均值
    2）计算帧间变化
    3）两路变化分别归一化
    4）加权融合
    5）平滑去抖

    Returns / 输出
    EN: `(S_smooth, dV_norm, dH_norm)`.
    中文：返回 `(平滑 S(t), 亮度变化序列, 色相变化序列)`。

    Mathematical intuition / 数学直觉
    EN:
    - `dV` captures photometric energy change
    - `dH` captures chromatic trajectory change on circular domain
    - `S(t)` is a weighted low-frequency proxy of visual activity
    中文：
    - `dV` 表示光度能量变化
    - `dH` 表示色彩轨迹变化（环形域）
    - `S(t)` 是视觉活动强度的加权低频代理信号
    """
    v_means: List[float] = []
    h_means: List[float] = []
    for frame in frames:
        # [EN] Per-frame brightness centroid on Value channel.
        # [中文] 每帧亮度通道的中心值（均值）。
        v_means.append(float(np.mean(frame[..., 2])))
        # [EN] Circular mean avoids hue wrap artifacts near 0/360.
        # [中文] 色相采用环形均值，避免 0/360 边界伪跳变。
        h_means.append(float(circular_mean_deg(frame[..., 0].reshape(-1))))

    dv = [0.0]
    dh = [0.0]
    for i in range(1, len(frames)):
        dv.append(abs(v_means[i] - v_means[i - 1]))
        # [EN] Normalize hue distance by 180 => range roughly [0,1].
        # [中文] 色相差除以 180 后归一到约 [0,1]。
        dh.append(circular_diff_deg(h_means[i], h_means[i - 1]) / 180.0)

    dv_n = normalize(dv)
    dh_n = normalize(dh)
    # [EN] Weighted fusion: brightness dynamics dominate, hue dynamics assist.
    # [中文] 加权融合：亮度动态为主，色相动态为辅。
    s_raw = [0.6 * dv_n[i] + 0.4 * dh_n[i] for i in range(len(dv_n))]
    # [EN] Final smoothing suppresses frame-level jitter before rhythm mapping.
    # [中文] 最终平滑用于抑制帧级抖动，再进入节奏映射。
    s_smooth = smooth_series(s_raw, smooth_window)
    return s_smooth, dv_n, dh_n


def segment_by_signal(signal: Sequence[float], min_len: int) -> List[Tuple[int, int]]:
    """Purpose / 目的
    EN: Build non-uniform temporal segments from S(t), used as harmonic structure.
    中文：根据 S(t) 构建非均匀时间分段，作为和声结构骨架。

    Method / 方法
    EN:
    - use low local minima as candidate boundaries
    - enforce minimum segment length
    - fallback to fixed-length chunks when minima are insufficient
    中文：
    - 使用低位局部谷值作为候选边界
    - 强制最小段长
    - 候选不足时退回固定长度分段

    Output semantics / 输出语义
    EN: each tuple is `[start_idx, end_idx)` in frame index space.
    中文：每个元组为帧索引区间 `[start_idx, end_idx)`。
    """
    n = len(signal)
    if n == 0:
        return []
    min_len = max(2, int(min_len))
    if n <= min_len:
        return [(0, n)]

    # [EN] 40th percentile acts as "low activity" boundary detector.
    # [中文] 40% 分位阈值用于检测“低活动区”边界。
    threshold = float(np.quantile(signal, 0.4))
    candidates = []
    for i in range(1, n - 1):
        if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1] and signal[i] <= threshold:
            candidates.append(i)

    # [EN] Always include sequence start and end as hard boundaries.
    # [中文] 序列起点与终点始终作为硬边界。
    boundaries = [0]
    last = 0
    for idx in candidates:
        if idx - last >= min_len:
            boundaries.append(idx)
            last = idx
    if n - boundaries[-1] < min_len and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(n)

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= 1:
            segments.append((start, end))
    # [EN] Degenerate case fallback: enforce fixed-length chunking.
    # [中文] 退化场景回退：强制固定长度切分。
    if len(segments) <= 1:
        segments = []
        for start in range(0, n, min_len):
            end = min(n, start + min_len)
            segments.append((start, end))
    return segments


def accent_strength(signal: Sequence[float]) -> List[float]:
    """Purpose / 目的
    EN: Derive sparse accent events from peaks of change-rate |dS/dt|.
    中文：从变化率 |dS/dt| 的峰值提取稀疏重音事件。

    Method / 方法
    EN: local-maximum detection + high-quantile thresholding.
    中文：局部极大值检测 + 高分位阈值筛选。

    Output / 输出
    EN: Accent strength array aligned to frame index.
    中文：与帧索引对齐的重音强度数组。

    Design rationale / 设计动机
    EN: accents should react to sudden changes, not just high absolute activity.
    中文：重音应响应“突变”，而不只是响应“持续高值”。
    """
    if not signal:
        return []
    d = [0.0]
    # [EN] First-order difference approximates |dS/dt| on discrete timeline.
    # [中文] 一阶差分近似离散时间下的 |dS/dt|。
    d.extend([abs(signal[i] - signal[i - 1]) for i in range(1, len(signal))])
    d_n = normalize(d)
    # [EN] High quantile keeps accents sparse and musically legible.
    # [中文] 高分位阈值可保持重音稀疏、音乐可读性更好。
    threshold = float(np.quantile(d_n, 0.85)) if len(d_n) > 5 else 0.5
    out = [0.0 for _ in d_n]
    for i in range(1, len(d_n) - 1):
        if d_n[i] >= d_n[i - 1] and d_n[i] >= d_n[i + 1] and d_n[i] >= threshold:
            out[i] = float(d_n[i])
    return out


def split_probability(mu: float, sigma: float) -> float:
    """Purpose / 目的
    EN: Convert global S(t) statistics into weak-beat split probability.
    中文：将 S(t) 全局统计转换为弱拍细分概率。

    Inputs / 输入
    EN: `mu` (mean activity), `sigma` (activity variance).
    中文：`mu`（活动均值）与 `sigma`（活动方差）。

    Notes / 说明
    EN: output is clamped to a safe interval to prevent extreme rhythm density.
    中文：输出做安全区间约束，防止节奏过密或过稀。

    Mapping intuition / 映射直觉
    EN:
    - high `mu`: globally active video => denser weak-beat subdivision
    - high `sigma`: unstable activity => higher transition tendency
    中文：
    - `mu` 高：视频整体活跃，弱拍细分倾向更强
    - `sigma` 高：活动波动大，节奏转移倾向更高
    """
    mu = float(max(0.0, min(1.0, mu)))
    # [EN] Empirical variance normalization constant tuned for observed S(t) scale.
    # [中文] 方差归一化常数按实际 S(t) 量级经验设定。
    sigma_n = min(1.0, max(0.0, sigma / 0.05))
    change_level = mu * 0.7 + sigma_n * 0.3
    return float(max(0.1, min(0.7, 0.1 + change_level * 0.6)))


def _sample_weak_depth(
    split_prob: float,
    max_depth: int,
    prev_depth: int,
    rng: np.random.Generator,
) -> int:
    """Purpose / 目的
    EN: Sample subdivision depth with temporal continuity constraints.
    中文：在时间连续性约束下采样细分深度。

    Method / 方法
    EN:
    - geometric sampling with decay for deeper levels
    - inertia constraint to limit abrupt depth jumps
    - isolated deepest split suppression to remove rhythmic flashes
    中文：
    - 几何衰减采样控制深层细分概率
    - 惯性约束限制相邻深度突变
    - 抑制孤立最深细分，减少“闪一下”听感

    Output / 输出
    EN: integer depth `d` where subdivisions = `2^d`.
    中文：返回整数深度 `d`，对应细分数为 `2^d`。
    """
    # [EN] Stable subdivision-depth policy:
    # [EN] 1) Geometric decay suppresses very deep splits.
    # [EN] 2) Inertia limits abrupt depth jumps between adjacent weak beats.
    # [中文] 稳定细分策略：
    # [中文] 1）通过几何衰减抑制过深细分概率。
    # [中文] 2）通过惯性约束限制相邻弱拍的深度突变。
    split_prob = float(max(0.0, min(1.0, split_prob)))
    depth = 0
    p = split_prob
    while depth < max_depth and rng.random() < p:
        depth += 1
        p *= 0.72

    # [EN] Suppress isolated deepest splits to avoid one-beat rhythmic flashes.
    # [中文] 抑制孤立最深细分，避免出现单拍突兀“闪动”。
    if depth >= 2 and prev_depth == 0 and rng.random() < 0.75:
        depth = 1

    # [EN] Inertia window limits one-step depth jump to ±1.
    # [中文] 惯性窗口把单步深度变化限制在 ±1。
    lo = max(0, prev_depth - 1)
    hi = min(max_depth, prev_depth + 1)
    return int(max(lo, min(hi, depth)))


def _emit_split(events: List[Tuple[float, float]], start: float, beat_len: float, depth: int) -> None:
    """EN: Emit equal subdivisions for one beat at the selected depth.
    中文：按给定深度把一个拍子等分并写入事件。

    EN: event tuple format is `(start_beat, duration_beat)`.
    中文：事件元组格式为 `(起始拍, 时值拍)`。
    """
    parts = 1 << int(max(0, depth))
    dur = beat_len / float(parts)
    for i in range(parts):
        events.append((start + i * dur, dur))


def generate_phrase_events(split_prob: float, rng: np.random.Generator, max_depth: int = 2) -> List[Tuple[float, float]]:
    """Purpose / 目的
    EN: Generate one reusable phrase-level rhythm pattern.
    中文：生成一个可循环复用的乐句级节奏模式。

    Grammar / 语法
    EN: 2 bars, each bar = Strong-Weak-Weak-Strong.
    中文：2 小节结构，每小节为 强-弱-弱-强。

    Constraints / 约束
    EN: strong beats are fixed; weak beats split by sampled depth up to `max_depth`.
    中文：强拍固定；弱拍按采样深度细分，深度不超过 `max_depth`。

    Determinism / 可复现性
    EN: For fixed seed and input stats, phrase generation is deterministic.
    中文：在固定种子与输入统计下，乐句生成可复现。
    """
    events: List[Tuple[float, float]] = []
    bar_structure = ("Strong", "Weak", "Weak", "Strong")
    prev_weak_depth = 0
    for bar_index in range(2):
        bar_start = bar_index * 4.0
        for beat_index, beat_type in enumerate(bar_structure):
            beat_start = bar_start + beat_index * 1.0
            if beat_type == "Strong":
                # [EN] Keep structural anchors stable for phrase intelligibility.
                # [中文] 强拍作为结构锚点，保持稳定以提升可听懂度。
                events.append((beat_start, 1.0))
            else:
                depth = _sample_weak_depth(split_prob, max_depth, prev_weak_depth, rng)
                _emit_split(events, beat_start, 1.0, depth)
                prev_weak_depth = depth
    events.sort(key=lambda e: e[0])
    return events


# ============================================================================
# [EN] MIDI and Synthesis
# [中文] MIDI 与合成渲染
# ============================================================================

def schedule_note(events: List[Tuple[int, mido.Message]], start: int, duration: int, note: int, velocity: int, channel: int) -> None:
    """EN: Append note_on + note_off using absolute tick time.
    中文：按绝对 tick 时间写入 note_on / note_off 成对事件。"""
    events.append((start, mido.Message("note_on", note=note, velocity=velocity, channel=channel)))
    events.append((start + duration, mido.Message("note_off", note=note, velocity=0, channel=channel)))


def midi_note_to_freq(note: int) -> float:
    """EN: Convert MIDI note number to frequency in Hz (A4=440).
    中文：将 MIDI 音高转换为赫兹频率（A4=440Hz）。"""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _adsr_envelope(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """EN: Generate an ADSR envelope with total length `n` samples.
    中文：生成长度为 `n` 的 ADSR 包络数组。"""
    # [EN] ADSR stage times are in seconds; release is applied to the note tail.
    # [中文] ADSR 各阶段参数单位为秒；释放段作用于音符尾部。
    attack_n = int(max(1, round(attack * sr)))
    decay_n = int(max(1, round(decay * sr)))
    release_n = int(max(1, round(release * sr)))
    sustain_n = max(0, n - attack_n - decay_n - release_n)

    env = np.zeros(n, dtype=np.float32)
    a_end = min(n, attack_n)
    if a_end > 0:
        env[:a_end] = np.linspace(0.0, 1.0, a_end, endpoint=False, dtype=np.float32)
    d_end = min(n, attack_n + decay_n)
    if d_end > attack_n:
        env[attack_n:d_end] = np.linspace(1.0, sustain, d_end - attack_n, endpoint=False, dtype=np.float32)
    s_end = min(n, attack_n + decay_n + sustain_n)
    if s_end > d_end:
        env[d_end:s_end] = sustain
    if n > s_end:
        env[s_end:] = np.linspace(sustain, 0.0, n - s_end, endpoint=True, dtype=np.float32)
    return env


def _waveform(phase: np.ndarray, kind: str) -> np.ndarray:
    """EN: Basic waveform generator from radian phase.
    中文：根据弧度相位生成基础波形。"""
    if kind == "sine":
        return np.sin(phase)
    if kind == "triangle":
        # [EN] Triangle waveform is derived from a transformed saw shape.
        # [中文] 三角波通过锯齿波形变换得到。
        saw = (phase / (2.0 * math.pi)) % 1.0
        tri = 2.0 * np.abs(2.0 * saw - 1.0) - 1.0
        return tri.astype(np.float32)
    if kind == "square":
        return np.sign(np.sin(phase)).astype(np.float32)
    # [EN] Fallback waveform path.
    # [中文] 默认回退波形路径。
    return np.sin(phase)


def _poly_blep(t01: np.ndarray, dt: float) -> np.ndarray:
    """EN: PolyBLEP correction curve for band-limiting sharp waveform edges.
    中文：PolyBLEP 校正曲线，用于抑制锯齿/方波边沿导致的混叠失真。"""
    # [EN] Apply PolyBLEP near discontinuities; t01 in [0,1), dt=freq/sr.
    # [中文] 在波形不连续点附近施加 PolyBLEP 校正；t01∈[0,1)，dt=freq/sr。
    if dt <= 0.0:
        return np.zeros_like(t01, dtype=np.float32)
    dt = float(min(0.5, dt))
    out = np.zeros_like(t01, dtype=np.float32)

    m = t01 < dt
    if np.any(m):
        x = (t01[m] / dt).astype(np.float32)
        out[m] = x + x - x * x - 1.0

    m = t01 > (1.0 - dt)
    if np.any(m):
        x = ((t01[m] - 1.0) / dt).astype(np.float32)
        out[m] = x * x + x + x + 1.0

    return out


def _oscillator(kind: str, freq: float, n: int, sample_rate: int, phase0: float = 0.0) -> np.ndarray:
    """EN: Oscillator core supporting sine/triangle/saw/square/pulse waveforms.
    中文：振荡器核心，支持正弦/三角/锯齿/方波/脉冲波。

    EN: Uses PolyBLEP-style correction for discontinuous waveforms to reduce aliasing.
    中文：对不连续波形应用 PolyBLEP 修正，降低混叠噪声。
    """
    # [EN] Anti-aliased oscillator core (pragmatic PolyBLEP variant).
    # [中文] 抗混叠振荡器核心（工程化 PolyBLEP 变体）。
    sr = float(max(8000, sample_rate))
    f = float(max(0.0, min(freq, 0.45 * sr)))
    if n <= 0 or f <= 0.0:
        return np.zeros(max(0, int(n)), dtype=np.float32)
    inc = f / sr
    t01 = (float(phase0) + inc * np.arange(int(n), dtype=np.float32)) % 1.0

    if kind == "sine":
        return np.sin(2.0 * math.pi * t01).astype(np.float32)
    if kind == "triangle":
        # [EN] Derive triangle waveform by integrating a BLEP-corrected square wave.
        # [中文] 通过对 BLEP 校正方波积分得到三角波。
        sq = np.where(t01 < 0.5, 1.0, -1.0).astype(np.float32)
        sq += _poly_blep(t01, inc) - _poly_blep((t01 + 0.5) % 1.0, inc)
        tri = np.cumsum(sq) * (2.0 * inc)
        tri = tri - np.mean(tri)
        peak = float(np.max(np.abs(tri))) if tri.size else 1.0
        return (tri / max(1e-6, peak)).astype(np.float32)
    if kind == "saw":
        saw = (2.0 * t01 - 1.0).astype(np.float32)
        saw -= _poly_blep(t01, inc)
        return saw.astype(np.float32)
    if kind == "square":
        sq = np.where(t01 < 0.5, 1.0, -1.0).astype(np.float32)
        sq += _poly_blep(t01, inc) - _poly_blep((t01 + 0.5) % 1.0, inc)
        return sq.astype(np.float32)
    if kind.startswith("pulse"):
        # [EN] Pulse waveform supports duty suffix, e.g. "pulse0.25".
        # [中文] 脉冲波支持占空比后缀，例如 "pulse0.25"。
        duty = 0.25
        if len(kind) > 5:
            try:
                duty = float(kind[5:])
            except Exception:
                duty = 0.25
        duty = float(max(0.05, min(0.95, duty)))
        y = np.where(t01 < duty, 1.0, -1.0).astype(np.float32)
        # [EN] Pulse waveform has two discontinuities per cycle (phase 0 and duty edge).
        # [中文] 脉冲波每周期有两个不连续点（相位 0 与 duty 边界）。
        y += _poly_blep(t01, inc) - _poly_blep((t01 - duty) % 1.0, inc)
        return y.astype(np.float32)
    # [EN] Fallback path.
    # [中文] 默认回退路径。
    return np.sin(2.0 * math.pi * t01).astype(np.float32)


def _sine_vibrato(freq: float, n: int, sample_rate: int, phase0: float, rate_hz: float, depth_cents: float) -> np.ndarray:
    """EN: Sine oscillator with vibrato (time-varying frequency in cents).
    中文：带颤音的正弦振荡器（以 cents 方式做时变频率）。"""
    # [EN] Vibrato is implemented on the sine path for numerical stability and low complexity.
    # [中文] 颤音仅在正弦路径实现，以兼顾数值稳定与复杂度控制。
    sr = float(max(8000, sample_rate))
    f0 = float(max(0.0, min(freq, 0.45 * sr)))
    if n <= 0 or f0 <= 0.0:
        return np.zeros(max(0, int(n)), dtype=np.float32)
    rate = float(max(0.0, rate_hz))
    depth = float(max(0.0, depth_cents))
    if rate <= 0.0 or depth <= 0.0:
        return _oscillator("sine", f0, n, sample_rate, phase0=phase0)

    t = (np.arange(int(n), dtype=np.float32) / float(sr)).astype(np.float32)
    cents = np.sin(2.0 * math.pi * rate * t).astype(np.float32) * float(depth)
    inst_f = f0 * (2.0 ** (cents / 1200.0)).astype(np.float32)
    phase = (float(phase0) + np.cumsum(inst_f / sr).astype(np.float32)) % 1.0
    return np.sin(2.0 * math.pi * phase).astype(np.float32)


def _additive_voice(
    freq: float,
    n: int,
    sample_rate: int,
    *,
    num_harmonics: int,
    harmonic_rolloff: float,
    harmonic_decay: float,
    harmonic_decay_slope: float,
    vib_rate_hz: float,
    vib_depth_cents: float,
    phase_seed: int,
) -> np.ndarray:
    """EN: Additive-synthesis voice built from harmonic partials.
    中文：基于谐波叠加的加法合成音色生成器。

    - harmonic_rolloff: amplitude ~ 1/k^rolloff
    - harmonic_decay: per-harmonic exponential decay base (0 => no decay)
    - harmonic_decay_slope: how much faster higher harmonics decay
    - vibrato is applied to the base frequency phase (cheap + musical).
    """
    sr = float(max(8000, sample_rate))
    f0 = float(max(0.0, min(freq, 0.45 * sr)))
    if n <= 0 or f0 <= 0.0:
        return np.zeros(max(0, int(n)), dtype=np.float32)

    nh = int(max(1, min(32, num_harmonics)))
    t = (np.arange(int(n), dtype=np.float32) / float(sr)).astype(np.float32)

    # [EN] Build instantaneous frequency trajectory (vibrato expressed in cents).
    # [中文] 构建瞬时频率轨迹（颤音以 cents 表示）。
    if vib_rate_hz > 0.0 and vib_depth_cents > 0.0:
        cents = (np.sin(2.0 * math.pi * float(vib_rate_hz) * t).astype(np.float32) * float(vib_depth_cents))
        inst_f = (f0 * (2.0 ** (cents / 1200.0)).astype(np.float32)).astype(np.float32)
    else:
        inst_f = np.full(int(n), f0, dtype=np.float32)

    # [EN] Integrate frequency trajectory to phase cycles.
    # [中文] 对频率轨迹积分得到相位（周期制）。
    phase_cycles = (np.cumsum(inst_f / float(sr)).astype(np.float32)) % 1.0

    rng = np.random.default_rng(int(phase_seed) & 0xFFFFFFFF)
    ph = rng.random(nh).astype(np.float32)

    out = np.zeros(int(n), dtype=np.float32)
    roll = float(max(0.0, harmonic_rolloff))
    d0 = float(max(0.0, harmonic_decay))
    ds = float(max(0.0, harmonic_decay_slope))
    for k in range(1, nh + 1):
        amp = 1.0 / (k**roll) if roll > 0.0 else 1.0
        sig = np.sin(2.0 * math.pi * (float(k) * phase_cycles + float(ph[k - 1]))).astype(np.float32)
        if d0 > 0.0:
            decay = np.exp(-t * (d0 * (k**ds))).astype(np.float32)
            sig *= decay
        out += sig * float(amp)

    peak = float(np.max(np.abs(out))) if out.size else 1.0
    if peak > 1e-6:
        out = out / peak
    return out.astype(np.float32)


def _velocity_curve(vel: int) -> float:
    """EN: Perceptual velocity mapping (MIDI 1..127 -> smooth loudness gain).
    中文：感知化力度映射（MIDI 力度到更平滑的响度增益）。"""
    # [EN] Velocity curve approximates perceptual loudness response.
    # [中文] 使用近似感知响度响应的力度曲线。
    v = float(max(1, min(127, int(vel)))) / 127.0
    return float(v**1.6)


def _soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """EN: Tanh soft-clipper for gentle saturation and peak control.
    中文：tanh 软削波，用于轻度饱和与峰值控制。"""
    d = float(max(0.01, drive))
    y = np.tanh(x * d) / math.tanh(d)
    return y.astype(np.float32)


def _next_pow2(n: int) -> int:
    """EN: Return next power-of-two size (FFT helper).
    中文：返回下一个 2 的幂（用于 FFT 尺寸）。"""
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """EN: 1D FFT convolution for long filters (e.g., convolution reverb).
    中文：一维 FFT 卷积，适用于长滤波核（如卷积混响）。"""
    # [EN] Use FFT convolution for long kernels; output length is len(x)+len(h)-1.
    # [中文] 长卷积核使用 FFT 卷积；输出长度为 len(x)+len(h)-1。
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    if x.size == 0 or h.size == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = int(x.size + h.size - 1)
    n_fft = _next_pow2(n_out)
    X = np.fft.rfft(x, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    y = np.fft.irfft(X * H, n=n_fft)[:n_out]
    return y.astype(np.float32)


def _make_reverb_ir(sample_rate: int, seconds: float, decay: float, lowpass_hz: float, seed: int) -> np.ndarray:
    """EN: Build a lightweight synthetic IR (early reflections + decaying late tail).
    中文：构造轻量算法 IR（早期反射 + 衰减噪声尾部）。"""
    sr = int(max(8000, sample_rate))
    n = int(max(1, round(seconds * sr)))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    t = (np.arange(n, dtype=np.float32) / float(sr)).astype(np.float32)

    # [EN] Model early reflections with sparse deterministic taps.
    # [中文] 早期反射使用稀疏确定性抽头建模。
    ir = np.zeros(n, dtype=np.float32)
    taps_s = [0.0, 0.018, 0.041, 0.073, 0.109, 0.157]
    taps_g = [1.0, 0.55, 0.38, 0.28, 0.20, 0.14]
    for ts, g in zip(taps_s, taps_g):
        idx = int(round(ts * sr))
        if 0 <= idx < n:
            ir[idx] += float(g)

    # [EN] Model late reverberation tail as exponentially decaying noise.
    # [中文] 后期混响尾部建模为指数衰减噪声。
    tail = rng.standard_normal(n).astype(np.float32)
    tail *= np.exp(-t * float(max(0.5, decay))).astype(np.float32)
    ir += 0.25 * tail

    # [EN] Apply gentle low-pass shaping in the frequency domain.
    # [中文] 在频域施加柔和的低通整形。
    fc = float(max(200.0, min(lowpass_hz, 0.45 * sr)))
    n_fft = _next_pow2(n)
    IR = np.fft.rfft(ir, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    lp = 1.0 / (1.0 + (freqs / fc) ** 4)
    out = np.fft.irfft(IR * lp, n=n_fft)[:n].astype(np.float32)

    peak = float(np.max(np.abs(out))) if out.size else 1.0
    if peak > 1e-6:
        out = out / peak
    return out.astype(np.float32)


def _apply_spectral_hp_lp(x: np.ndarray, sample_rate: int, hp_hz: float, lp_hz: float) -> np.ndarray:
    """EN: Apply broad high/low-pass spectral shaping in frequency domain.
    中文：在频域施加平滑高通+低通，进行全局音色整形。"""
    # [EN] Lightweight global tonal shaping with smooth high-pass and low-pass filters.
    # [中文] 采用轻量级全局音色整形：平滑高通与低通。
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    sr = float(max(8000, sample_rate))
    n = int(x.size)
    n_fft = _next_pow2(n)
    X = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    hp = float(max(5.0, hp_hz))
    lp = float(max(hp + 50.0, lp_hz))
    H_hp = 1.0 - (1.0 / (1.0 + (freqs / hp) ** 4))
    H_lp = 1.0 / (1.0 + (freqs / lp) ** 4)
    y = np.fft.irfft(X * (H_hp * H_lp), n=n_fft)[:n]
    return y.astype(np.float32)


def _one_pole_lp_varcut(x: np.ndarray, cutoff_hz: np.ndarray, sample_rate: int, poles: int = 2) -> np.ndarray:
    """EN: Time-varying low-pass filter for per-note cutoff envelope motion.
    中文：时变低通滤波器，用于每个音符的 cutoff 包络运动。"""
    # [EN] Time-varying one-pole low-pass filter cascaded `poles` times.
    # [中文] 时变一阶低通串联 `poles` 次以增强截止斜率。
    x = np.asarray(x, dtype=np.float32)
    cutoff_hz = np.asarray(cutoff_hz, dtype=np.float32)
    if x.size == 0:
        return x
    sr = float(max(8000, sample_rate))
    if cutoff_hz.size != x.size:
        cutoff_hz = np.resize(cutoff_hz, x.shape).astype(np.float32)

    # [EN] Convert cutoff frequency to one-pole smoothing coefficients.
    # [中文] 将 cutoff 频率转换为一阶滤波平滑系数。
    fc = np.clip(cutoff_hz, 30.0, 0.45 * sr).astype(np.float32)
    a = (1.0 - np.exp(-2.0 * math.pi * fc / sr)).astype(np.float32)

    y = x
    for _ in range(int(max(1, poles))):
        out = np.empty_like(y)
        last = float(y[0])
        out[0] = last
        # [EN] Use an explicit loop to prioritize numerical predictability over vectorized complexity.
        # [中文] 使用显式循环以优先保证数值可预测性与可解释性。
        for i in range(1, y.size):
            ai = float(a[i])
            last = last + ai * (float(y[i]) - last)
            out[i] = last
        y = out
    return y.astype(np.float32)


def collect_note_intervals(events: Sequence[Tuple[int, mido.Message]]) -> List[Tuple[int, int, int, int]]:
    """EN: Convert raw MIDI-like messages into note intervals.
    中文：将原始 MIDI 风格事件整理为“起止区间音符”。

    EN output tuple:
    (start_tick, end_tick, midi_note, velocity)
    中文输出元组：
    (start_tick, end_tick, midi_note, velocity)
    """
    # [EN] Tuple format: (start_tick, end_tick, midi_note, velocity)
    # [中文] 元组格式：(起始 tick, 结束 tick, 音高, 力度)
    ordered = sorted(events, key=lambda e: e[0])
    active: dict[Tuple[int, int], Tuple[int, int]] = {}
    notes: List[Tuple[int, int, int, int]] = []
    for t, msg in ordered:
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (int(getattr(msg, "channel", 0)), int(getattr(msg, "note", 0)))
        if msg.type == "note_on" and int(msg.velocity) > 0:
            active[key] = (int(t), int(msg.velocity))
        else:
            if key in active:
                start, vel = active.pop(key)
                end = int(t)
                if end > start:
                    notes.append((start, end, key[1], vel))
    return notes


def synthesize_to_wav(
    path: str,
    tempo_bpm: int,
    ticks_per_beat: int,
    sample_rate: int,
    harmony_events: Sequence[Tuple[int, mido.Message]],
    bass_events: Sequence[Tuple[int, mido.Message]],
    melody_events: Sequence[Tuple[int, mido.Message]],
    pad_events: Sequence[Tuple[int, mido.Message]],
) -> None:
    """Purpose / 目的
    EN: Render Harmony/Bass/Melody/Pad event streams into stereo PCM WAV.
    中文：将和声/低音/旋律/铺底四轨事件渲染为立体声 PCM WAV。

    Inputs / 输入
    EN: tempo/timing parameters + per-track note events in absolute MIDI ticks.
    中文：速度与时间参数 + 各轨绝对 tick 音符事件。

    Signal Path / 信号路径
    EN (per note):
    source -> bright layer -> transient -> filter envelope -> saturation -> ADSR
    EN (bus):
    dry mix -> convolution reverb -> soft limiting/normalization
    中文（单音）：
    音源 -> 亮度层 -> 起音瞬态 -> 滤波包络 -> 饱和 -> ADSR
    中文（总线）：
    干声混合 -> 卷积混响 -> 软限制/归一化
    """
    seconds_per_tick = (60.0 / max(1, tempo_bpm)) / float(ticks_per_beat)

    harmony_notes = collect_note_intervals(harmony_events)
    bass_notes = collect_note_intervals(bass_events)
    melody_notes = collect_note_intervals(melody_events)
    pad_notes = collect_note_intervals(pad_events)

    all_notes = harmony_notes + bass_notes + melody_notes + pad_notes
    if not all_notes:
        # [EN] If no note events exist, emit a short silent reference file.
        # [中文] 若无音符事件，则输出一个短静音参考文件。
        n = int(sample_rate * 2.0)
        audio = np.zeros((n, 2), dtype=np.float32)
    else:
        end_tick = max(n[1] for n in all_notes)
        length_s = end_tick * seconds_per_tick + 0.5
        n = int(max(1, round(length_s * sample_rate)))

        # [EN] Maintain a dry stereo bus plus a mono reverb send for efficient processing.
        # [中文] 使用干声立体声总线与单声道混响发送，兼顾效率与空间感。
        audio = np.zeros((n, 2), dtype=np.float32)
        send = np.zeros(n, dtype=np.float32)

        def add_notes(
            notes: List[Tuple[int, int, int, int]],
            *,
            body_kind: str,
            bright_kind: str,
            detune_cents: Sequence[float],
            harmonics: Sequence[Tuple[float, float]],
            gain: float,
            pan: float,
            adsr_amp: Tuple[float, float, float, float],
            adsr_bright: Tuple[float, float, float, float],
            bright_gain: float,
            sat_drive: float,
            reverb_send: float,
            noise_gain: float,
            noise_adsr: Tuple[float, float, float, float],
            sine_vib_rate_hz: float,
            sine_vib_depth_cents: float,
            filter_base_hz: float,
            filter_env_hz: float,
            filter_adsr: Tuple[float, float, float, float],
            filter_keytrack: float,
            filter_poles: int,
        ) -> None:
            # [EN] Use constant-power panning to avoid apparent loudness collapse at center position.
            # [中文] 使用恒功率声像律，减少中置时听感音量塌陷。
            left = math.cos((pan + 1.0) * math.pi / 4.0)
            right = math.sin((pan + 1.0) * math.pi / 4.0)
            detune = list(detune_cents) if detune_cents else [0.0]
            harms = list(harmonics) if harmonics else []

            for start_tick, end_tick, note, vel in notes:
                # [EN] Convert MIDI-tick timing into sample-index timing.
                # [中文] 将 MIDI tick 时间线映射到音频采样索引时间线。
                start = int(round(start_tick * seconds_per_tick * sample_rate))
                end = int(round(end_tick * seconds_per_tick * sample_rate))
                if end <= start:
                    continue
                end = min(end, n)
                start = max(0, min(start, n - 1))
                seg_n = end - start
                if seg_n <= 0:
                    continue

                freq = midi_note_to_freq(note)
                amp = _velocity_curve(vel) * float(gain)

                # [EN] Use independent envelopes for amplitude, brightness, and cutoff movement.
                # [中文] 振幅、亮度与截止频率移动分别由独立包络控制。
                env_amp = _adsr_envelope(seg_n, sample_rate, *adsr_amp).astype(np.float32)
                # [EN] Brightness envelope controls the crossfade of the bright layer over time.
                # [中文] 亮度包络用于控制高频亮层的时间交叉渐变。
                env_bright = _adsr_envelope(seg_n, sample_rate, *adsr_bright).astype(np.float32)
                env_filt = _adsr_envelope(seg_n, sample_rate, *filter_adsr).astype(np.float32)

                body = np.zeros(seg_n, dtype=np.float32)
                bright = np.zeros(seg_n, dtype=np.float32)
                for i, cents in enumerate(detune):
                    f = freq * (2.0 ** (float(cents) / 1200.0))
                    phase0 = float((i + 1) * 0.173) % 1.0
                    # [EN] The body source can be either a classic oscillator or an additive instrument model.
                    # [中文] 主体音源可选传统振荡器，或加法式“乐器化”模型。
                    if body_kind == "additive_string":
                        body += _additive_voice(
                            f,
                            seg_n,
                            sample_rate,
                            num_harmonics=14,
                            harmonic_rolloff=1.15,
                            harmonic_decay=0.0,
                            harmonic_decay_slope=1.0,
                            vib_rate_hz=float(sine_vib_rate_hz),
                            vib_depth_cents=float(sine_vib_depth_cents),
                            phase_seed=int((note * 1315423911 + start_tick) & 0xFFFFFFFF),
                        )
                    elif body_kind == "additive_pluck":
                        body += _additive_voice(
                            f,
                            seg_n,
                            sample_rate,
                            num_harmonics=18,
                            harmonic_rolloff=0.95,
                            harmonic_decay=10.5,
                            harmonic_decay_slope=0.85,
                            vib_rate_hz=0.0,
                            vib_depth_cents=0.0,
                            phase_seed=int((note * 2654435761 + start_tick) & 0xFFFFFFFF),
                        )
                    else:
                        body += _oscillator(body_kind, f, seg_n, sample_rate, phase0=phase0)

                    # [EN] The bright layer is decoupled so timbre can respond independently to velocity.
                    # [中文] 将亮层独立混合，使音色可独立响应力度变化。
                    if bright_kind != "none":
                        if bright_kind == "sine" and (sine_vib_rate_hz > 0.0 and sine_vib_depth_cents > 0.0):
                            bright += _sine_vibrato(
                                f,
                                seg_n,
                                sample_rate,
                                phase0=(phase0 + 0.37) % 1.0,
                                rate_hz=float(sine_vib_rate_hz),
                                depth_cents=float(sine_vib_depth_cents),
                            )
                        else:
                            bright += _oscillator(bright_kind, f, seg_n, sample_rate, phase0=(phase0 + 0.37) % 1.0)
                body /= float(len(detune))
                if bright_kind != "none":
                    bright /= float(len(detune))

                # [EN] Add an extra harmonic layer to strengthen timbral identity.
                # [中文] 附加谐波层用于增强音色辨识度。
                if harms:
                    hsig = np.zeros(seg_n, dtype=np.float32)
                    for ratio, hg in harms:
                        r = float(max(0.25, min(8.0, ratio)))
                        g = float(max(0.0, min(2.0, hg)))
                        hsig += _oscillator("sine", freq * r, seg_n, sample_rate, phase0=0.11) * g
                    body = (body + 0.35 * hsig).astype(np.float32)

                vel01 = float(max(1, min(127, int(vel)))) / 127.0
                bg = float(bright_gain) * (0.55 + 0.75 * vel01)
                sig = body.astype(np.float32)
                if bright_kind != "none":
                    sig = (sig + bg * bright * env_bright).astype(np.float32)

                # [EN] Add short transient noise to emulate onset energy (pluck/bow/hammer).
                # [中文] 通过短瞬态噪声模拟拨弦/拉弓/击弦的起音能量。
                ng = float(max(0.0, noise_gain))
                if ng > 1e-6:
                    rng = np.random.default_rng(int((note * 1315423911 + start_tick * 2654435761) & 0xFFFFFFFF))
                    noise = rng.standard_normal(seg_n).astype(np.float32)
                    noise_env = _adsr_envelope(seg_n, sample_rate, *noise_adsr).astype(np.float32)
                    sig += noise * noise_env * ng

                # [EN] True cutoff-envelope behavior: low-pass cutoff follows a time-varying contour.
                # [中文] 真实截止频率包络：低通截止频率随时间轮廓变化。
                base = float(max(60.0, filter_base_hz))
                env_amt = float(max(0.0, filter_env_hz)) * (0.55 + 0.75 * vel01)
                kt = float(max(0.0, filter_keytrack))
                cutoff = (base + env_amt * env_filt + kt * float(freq)).astype(np.float32)
                sig = _one_pole_lp_varcut(sig, cutoff, sample_rate, poles=int(filter_poles))

                sig = _soft_clip(sig, sat_drive)
                sig = (sig * env_amp * amp).astype(np.float32)

                audio[start:end, 0] += sig * left
                audio[start:end, 1] += sig * right
                if reverb_send > 1e-6:
                    # [EN] Use mono reverb send to reduce convolution cost with minimal perceptual loss.
                    # [中文] 使用单声道混响发送以降低卷积开销，并保持听感损失最小。
                    send[start:end] += sig * float(reverb_send) * 0.5 * float(left + right)

        # [EN] Per-track synthesis presets are timbre-oriented and classroom-demo friendly.
        # [中文] 分轨合成预设以音色区分为核心，适合课堂演示。
        add_notes(
            harmony_notes,
            # [EN] Piano-like layer.
            # [中文] 钢琴感音色层。
            body_kind="additive_pluck",
            bright_kind="pulse0.35",
            detune_cents=(0.0,),
            harmonics=(),
            gain=0.30,
            pan=-0.10,
            adsr_amp=(0.002, 0.28, 0.18, 0.22),
            adsr_bright=(0.001, 0.06, 0.00, 0.05),
            bright_gain=0.50,
            sat_drive=1.35,
            reverb_send=0.10,
            noise_gain=0.085,
            noise_adsr=(0.001, 0.05, 0.00, 0.03),
            sine_vib_rate_hz=0.0,
            sine_vib_depth_cents=0.0,
            filter_base_hz=1200.0,
            filter_env_hz=5200.0,
            filter_adsr=(0.001, 0.22, 0.10, 0.18),
            filter_keytrack=0.70,
            filter_poles=2,
        )
        add_notes(
            bass_notes,
            body_kind="sine",
            bright_kind="triangle",
            detune_cents=(0.0,),
            harmonics=((2.0, 0.18), (0.5, 0.08)),
            gain=0.45,
            pan=-0.20,
            adsr_amp=(0.005, 0.12, 0.78, 0.10),
            adsr_bright=(0.001, 0.08, 0.00, 0.05),
            bright_gain=0.18,
            sat_drive=2.4,
            reverb_send=0.02,
            noise_gain=0.0,
            noise_adsr=(0.001, 0.02, 0.00, 0.02),
            sine_vib_rate_hz=0.0,
            sine_vib_depth_cents=0.0,
            filter_base_hz=180.0,
            filter_env_hz=520.0,
            filter_adsr=(0.001, 0.10, 0.20, 0.10),
            filter_keytrack=0.08,
            filter_poles=2,
        )
        add_notes(
            pad_notes,
            # [EN] Violin-like sustained pad layer.
            # [中文] 小提琴感持续铺底层。
            body_kind="additive_string",
            bright_kind="none",
            detune_cents=(-4.0, 4.0),
            harmonics=((2.0, 0.10), (3.0, 0.05)),
            gain=0.18,
            pan=0.15,
            adsr_amp=(0.10, 0.40, 0.92, 0.85),
            adsr_bright=(0.05, 0.40, 0.30, 0.60),
            bright_gain=0.0,
            sat_drive=1.10,
            reverb_send=0.30,
            noise_gain=0.030,
            noise_adsr=(0.010, 0.40, 0.20, 0.70),
            sine_vib_rate_hz=5.2,
            sine_vib_depth_cents=18.0,
            filter_base_hz=700.0,
            filter_env_hz=2100.0,
            filter_adsr=(0.06, 0.60, 0.55, 0.90),
            filter_keytrack=0.30,
            filter_poles=2,
        )
        add_notes(
            melody_notes,
            # [EN] Guzheng-like bright pluck layer.
            # [中文] 古筝感明亮拨弦层。
            body_kind="additive_pluck",
            bright_kind="pulse0.20",
            detune_cents=(0.0,),
            harmonics=((2.0, 0.14), (3.0, 0.06), (4.0, 0.03)),
            gain=0.30,
            pan=0.22,
            adsr_amp=(0.001, 0.18, 0.05, 0.12),
            adsr_bright=(0.001, 0.06, 0.00, 0.05),
            bright_gain=0.70,
            sat_drive=1.40,
            reverb_send=0.14,
            noise_gain=0.055,
            noise_adsr=(0.001, 0.04, 0.00, 0.03),
            sine_vib_rate_hz=0.0,
            sine_vib_depth_cents=0.0,
            filter_base_hz=1600.0,
            filter_env_hz=6500.0,
            filter_adsr=(0.001, 0.10, 0.08, 0.10),
            filter_keytrack=0.85,
            filter_poles=2,
        )

        # [EN] Perform global tonal cleanup before reverb (remove rumble and harsh high end).
        # [中文] 混响前执行全局音色清理（去除低频隆隆与高频刺耳）。
        audio[:, 0] = _apply_spectral_hp_lp(audio[:, 0], sample_rate, hp_hz=28.0, lp_hz=14000.0)
        audio[:, 1] = _apply_spectral_hp_lp(audio[:, 1], sample_rate, hp_hz=28.0, lp_hz=14000.0)

        # [EN] Use convolution reverb with short synthetic IRs (decorrelated left/right).
        # [中文] 使用短 IR 卷积混响，左右声道采用不同种子去相关。
        if np.any(send):
            ir_l = _make_reverb_ir(sample_rate, seconds=1.25, decay=1.55, lowpass_hz=8000.0, seed=1337)
            ir_r = _make_reverb_ir(sample_rate, seconds=1.25, decay=1.55, lowpass_hz=7600.0, seed=7331)
            wet_l = _fft_convolve_1d(send, ir_l)
            wet_r = _fft_convolve_1d(send, ir_r)

            out_n = int(max(audio.shape[0], wet_l.size, wet_r.size))
            if out_n != audio.shape[0]:
                pad = np.zeros((out_n, 2), dtype=np.float32)
                pad[: audio.shape[0], :] = audio
                audio = pad
            if wet_l.size < out_n:
                wet_l = np.pad(wet_l, (0, out_n - wet_l.size)).astype(np.float32)
            if wet_r.size < out_n:
                wet_r = np.pad(wet_r, (0, out_n - wet_r.size)).astype(np.float32)

            wet_gain = 0.22
            audio[:, 0] += wet_l * wet_gain
            audio[:, 1] += wet_r * wet_gain

        # [EN] Final peak handling and normalization for robust export loudness.
        # [中文] 最终执行峰值控制与归一化，保证导出响度稳定。
        audio = _soft_clip(audio, drive=2.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = audio / max(1.0, peak * 1.02)

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def render_midi_to_wav_with_fluidsynth(
    midi_path: str,
    wav_path: str,
    soundfont_path: str,
    sample_rate: int,
    gain: float,
) -> None:
    """Purpose / 目的
    EN: Offline render MIDI to WAV through fluidsynth with an SF2 library.
    中文：使用 fluidsynth + SF2 音色库离线渲染 MIDI 为 WAV。

    Notes / 说明
    EN: Requires local `fluidsynth` binary and a valid `.sf2` file path.
    中文：依赖本地 `fluidsynth` 可执行文件与有效 `.sf2` 路径。
    """
    fs = shutil.which("fluidsynth")
    if not fs:
        raise RuntimeError(
            "fluidsynth 未安装，无法通过 SoundFont 渲染音频。\\n"
            "macOS 安装：brew install fluid-synth\\n"
            "然后重试，并提供 --soundfont /path/to.sf2"
        )
    if not soundfont_path or not os.path.exists(soundfont_path):
        raise RuntimeError("SoundFont 文件不存在，请传入 --soundfont /path/to.sf2")

    sr = int(max(8000, sample_rate))
    g = float(max(0.0, min(gain, 5.0)))
    cmd = [
        fs,
        "-ni",
        "-g",
        str(g),
        "-r",
        str(sr),
        "-F",
        wav_path,
        soundfont_path,
        midi_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fluidsynth 渲染失败（code={proc.returncode}）。\\n{proc.stderr.strip()}")


def write_temp_midi_for_render(
    path: str,
    tempo_bpm: int,
    ticks_per_beat: int,
    tracks: Sequence[Tuple[str, List[Tuple[int, mido.Message]]]],
) -> None:
    """EN: Build temporary Type-1 MIDI file as fluidsynth render input.
    中文：生成临时 Type-1 MIDI，作为 fluidsynth 渲染输入。"""
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    midi.tracks.append(tempo_track)

    for name, events in tracks:
        tr = render_track(events)
        tr = prepend_meta(tr, [mido.MetaMessage("track_name", name=name)])
        midi.tracks.append(tr)

    midi.save(path)


def render_track(events: List[Tuple[int, mido.Message]]) -> mido.MidiTrack:
    """EN: Convert absolute-tick events to MIDI delta-time encoding.
    中文：把绝对 tick 事件转换为 MIDI 的 delta-time 轨道格式。"""
    events.sort(key=lambda e: e[0])
    track = mido.MidiTrack()
    last = 0
    for t, msg in events:
        delta = t - last
        msg.time = max(0, int(delta))
        track.append(msg)
        last = t
    return track


def clamp_note(note: int, low: int, high: int) -> int:
    """EN: Clamp MIDI pitch into allowed register range.
    中文：将 MIDI 音高限制在允许音区内。"""
    return int(max(low, min(high, note)))


def clamp_int(value: int, low: int, high: int) -> int:
    """EN: Generic integer clamp helper.
    中文：通用整数夹取函数。"""
    return int(max(low, min(high, value)))


def prepend_meta(track: mido.MidiTrack, metas: Sequence[mido.MetaMessage]) -> mido.MidiTrack:
    """EN: Prepend tempo/name/meta events at track start.
    中文：在轨道开头插入速度/名称等元事件。"""
    for meta in reversed(list(metas)):
        meta.time = 0
        track.insert(0, meta)
    return track


def add_gm_setup(events: List[Tuple[int, mido.Message]], channel: int, program: int, volume: int = 100, pan: int = 64) -> None:
    """EN: Append standard GM setup (bank/program/volume/pan) for one channel.
    中文：为单个通道追加标准 GM 初始化（音色库/音量/声像等）。"""
    # [EN] Standard GM initialization sequence: bank, volume, pan, then program change.
    # [中文] 标准 GM 初始化顺序：Bank、音量、声像、再到音色号。
    events.append((0, mido.Message("control_change", control=0, value=0, channel=channel)))
    events.append((0, mido.Message("control_change", control=32, value=0, channel=channel)))
    events.append((0, mido.Message("control_change", control=7, value=volume, channel=channel)))
    events.append((0, mido.Message("control_change", control=10, value=pan, channel=channel)))
    events.append((0, mido.Message("program_change", program=program, channel=channel)))


def remap_events_channel(events: List[Tuple[int, mido.Message]], channel: int) -> List[Tuple[int, mido.Message]]:
    """EN: Remap musical note events to a target channel (ignore setup events).
    中文：将音乐事件重映射到目标通道（跳过初始化类事件）。"""
    remapped: List[Tuple[int, mido.Message]] = []
    for t, msg in events:
        if msg.type in ("program_change", "control_change"):
            continue
        if hasattr(msg, "channel"):
            remapped.append((t, msg.copy(channel=channel)))
        else:
            remapped.append((t, msg.copy()))
    return remapped


def write_single_track_midi(
    path: str,
    events: List[Tuple[int, mido.Message]],
    tempo_bpm: int,
    track_name: str,
    instrument_name: str,
    program: int,
    auto_instrument: bool,
) -> None:
    """EN: Export one part as standalone Type-0 MIDI for DAW import.
    中文：导出单轨 Type-0 MIDI，便于 DAW 独立导入。"""
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    events_ch = remap_events_channel(events, 0)
    setup: List[Tuple[int, mido.Message]] = []
    if auto_instrument:
        add_gm_setup(setup, 0, program=program, volume=100, pan=64)
    else:
        setup.append((0, mido.Message("program_change", program=program, channel=0)))
    merged = setup + events_ch
    track = render_track(merged)
    track = prepend_meta(
        track,
        [
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)),
            mido.MetaMessage("track_name", name=track_name),
            mido.MetaMessage("instrument_name", name=instrument_name),
        ],
    )
    midi.tracks.append(track)
    midi.save(path)


def write_logic_bundle(
    output_dir: str,
    tempo_bpm: int,
    harmony_events: List[Tuple[int, mido.Message]],
    bass_events: List[Tuple[int, mido.Message]],
    melody_events: List[Tuple[int, mido.Message]],
    pad_events: List[Tuple[int, mido.Message]],
    auto_instrument: bool,
) -> None:
    """EN: Export Logic-friendly package (per-track MIDIs + import note).
    中文：导出 Logic 友好工程包（分轨 MIDI + 导入说明）。"""
    os.makedirs(output_dir, exist_ok=True)
    write_single_track_midi(
        os.path.join(output_dir, "Harmony.mid"),
        harmony_events,
        tempo_bpm,
        "Harmony",
        "Harmony",
        0,
        auto_instrument,
    )
    write_single_track_midi(
        os.path.join(output_dir, "Bass.mid"),
        bass_events,
        tempo_bpm,
        "Bass",
        "Bass",
        32,
        auto_instrument,
    )
    write_single_track_midi(
        os.path.join(output_dir, "Melody.mid"),
        melody_events,
        tempo_bpm,
        "Melody",
        "Melody",
        73,
        auto_instrument,
    )
    write_single_track_midi(
        os.path.join(output_dir, "Pad.mid"),
        pad_events,
        tempo_bpm,
        "Pad",
        "Pad",
        48,
        auto_instrument,
    )

    instructions = (
        "Logic Pro 导入建议：\\n"
        "1) File -> Import -> MIDI File...\\n"
        "2) 选择 “Create New Tracks”\\n"
        "3) 每个轨道选择一个 Software Instrument\\n"
        "提示：如果使用 GM 多音色音源，可自动响应 Program Change。\\n"
    )
    with open(os.path.join(output_dir, "IMPORT_IN_LOGIC.md"), "w", encoding="utf-8") as f:
        f.write(instructions)


def generate_midi(
    segments: Sequence[SegmentInfo],
    accent: Sequence[float],
    fps: float,
    total_frames: int,
    mu: float,
    sigma: float,
    seed: int,
    output: str,
    tempo_bpm: int,
    bars_per_segment: int,
    scale_name: str,
    single_channel: bool,
    logic_pro: bool,
    auto_instrument: bool,
    sensitivity: float,
    mapping_bias: float,
    mood: Optional[str],
    logic_structure: bool,
    direct_output: bool,
    synth_output: bool,
    sample_rate: int,
    soundfont: Optional[str],
    fluidsynth_gain: float,
) -> None:
    """Purpose / 目的
    EN: Translate visual features and rhythm controls into musical events.
    中文：把视觉特征与节奏控制参数转换为可播放音乐事件。

    Inputs / 输入
    EN: segment features, accent curve, FPS/timeline stats, style controls.
    中文：分段特征、重音曲线、时间轴参数与风格控制参数。

    Mapping Layers / 映射层级
    EN:
    1) Global: tempo/scale/mood
    2) Segment: harmony, bass, pad sustained by segment
    3) Phrase: melody rhythm from grammar + S(t) statistics
    4) Accent: local velocity emphasis
    中文：
    1）全局层：速度、调式、情绪
    2）分段层：按段持续的和声/低音/铺底
    3）乐句层：由语法 + S(t) 统计驱动的旋律节奏
    4）重音层：局部力度强化

    Outputs / 输出
    EN: Depending on mode, write MIDI, built-in synth WAV, or SoundFont-rendered WAV.
    中文：按模式输出 MIDI、内置合成 WAV 或 SoundFont 渲染 WAV。
    """
    ticks_per_beat = 480
    # [EN] SoundFont mode requires separate channels so each part can keep an independent program.
    # [中文] SoundFont 模式需要分通道，以保证各分轨可独立指定音色。
    if direct_output:
        single_channel = False
    if auto_instrument:
        single_channel = False
        logic_pro = True
    if logic_pro:
        single_channel = False
    midi_type = 0 if single_channel else 1
    midi = mido.MidiFile(type=midi_type, ticks_per_beat=ticks_per_beat)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    if auto_instrument:
        tempo_track.append(mido.Message("sysex", data=[0x7E, 0x7F, 0x09, 0x01], time=0))  # [EN] GM System On SysEx. [中文] GM System On 系统消息。
    midi.tracks.append(tempo_track)

    scale = SCALES[scale_name]
    mood_settings = MOODS.get(mood, {}) if mood else {}
    tensions = normalize([s.features.f_param_mean for s in segments])
    tension_factor = float(mood_settings.get("tension_factor", 1.0))

    harmony_events: List[Tuple[int, mido.Message]] = []
    bass_events: List[Tuple[int, mido.Message]] = []
    melody_events: List[Tuple[int, mido.Message]] = []
    pad_events: List[Tuple[int, mido.Message]] = []

    # [EN] General MIDI channel layout for four musical roles.
    # [中文] 四个音乐角色对应的 General MIDI 通道布局。
    ch_harmony = 0
    ch_bass = 1
    ch_pad = 2
    ch_melody = 3
    if single_channel:
        ch_bass = ch_pad = ch_melody = ch_harmony

    # [EN] In single-channel mode, avoid conflicting multi-program assignments.
    # [中文] 单通道模式下需避免多音色设置冲突。
    if auto_instrument:
        add_gm_setup(harmony_events, ch_harmony, program=0, volume=100, pan=64)   # [EN] Acoustic Grand Piano. [中文] 原声大钢琴。
        add_gm_setup(bass_events, ch_bass, program=32, volume=110, pan=48)       # [EN] Acoustic Bass. [中文] 原声低音提琴。
        add_gm_setup(pad_events, ch_pad, program=48, volume=90, pan=80)          # [EN] String Ensemble. [中文] 弦乐合奏。
        add_gm_setup(melody_events, ch_melody, program=73, volume=105, pan=64)   # [EN] Flute. [中文] 长笛。
    else:
        harmony_events.append((0, mido.Message("program_change", program=0, channel=ch_harmony)))  # [EN] Acoustic Grand Piano. [中文] 原声大钢琴。
        if not single_channel:
            bass_events.append((0, mido.Message("program_change", program=32, channel=ch_bass)))       # [EN] Acoustic Bass. [中文] 原声低音提琴。
            pad_events.append((0, mido.Message("program_change", program=48, channel=ch_pad)))         # [EN] String Ensemble. [中文] 弦乐合奏。
            melody_events.append((0, mido.Message("program_change", program=73, channel=ch_melody)))   # [EN] Flute. [中文] 长笛。

    ticks_per_second = ticks_per_beat * (tempo_bpm / 60.0)
    ticks_per_frame = ticks_per_second / max(1e-6, fps)
    total_ticks = int(round(total_frames * ticks_per_frame * max(1, bars_per_segment)))
    total_ticks = int(max(1, total_ticks))
    global_activity = _clamp01(mu + sigma * 2.5)
    global_valence, global_arousal = global_affect(segments)
    _, global_sync, _ = rhythm_targets_from_affect(global_valence, global_arousal, global_activity)
    melody_rhythm_steps_per_beat = choose_global_rhythm_steps_per_beat(global_activity, global_sync)
    # [EN] Harmony grid supports 1/8 + 1/6 union; bass/pad stay on quarter-note grid.
    # [中文] 和声网格采用 1/8 + 1/6 并集；低音与铺底保持四分网格。
    harmony_rhythm_steps_per_beat: Tuple[int, int] = (2, 6)
    bass_pad_rhythm_steps_per_beat = 1
    # [EN] Phrase repetition policy: low activity -> 4 bars, otherwise 2 bars.
    # [中文] 乐句重复策略：低活动用 4 小节，否则 2 小节。
    phrase_beats = 16.0 if global_activity < 0.45 else 8.0

    segment_plans = build_segment_module_plans(
        segments=segments,
        scale=scale,
        tensions=tensions,
        tension_factor=tension_factor,
        sensitivity=sensitivity,
        mood_settings=mood_settings,
        ticks_per_frame=ticks_per_frame,
        bars_per_segment=bars_per_segment,
        ticks_per_beat=ticks_per_beat,
        rhythm_steps_per_beat=harmony_rhythm_steps_per_beat,
    )
    harmonic_plans = adapt_harmonic_progression_rate(
        segment_plans,
        ticks_per_beat=ticks_per_beat,
        rhythm_steps_per_beat=harmony_rhythm_steps_per_beat,
    )
    emit_harmony_module(harmony_events, harmonic_plans, channel=ch_harmony)
    emit_bass_module(bass_events, harmonic_plans, channel=ch_bass)
    emit_pad_module(pad_events, harmonic_plans, channel=ch_pad)

    emit_melody_module(
        melody_events,
        segments=segments,
        accent=accent,
        harmony_events=harmony_events,
        bass_events=bass_events,
        pad_events=pad_events,
        total_frames=total_frames,
        ticks_per_frame=ticks_per_frame,
        total_ticks=total_ticks,
        ticks_per_beat=ticks_per_beat,
        phrase_beats=phrase_beats,
        seed=seed,
        mu=mu,
        sigma=sigma,
        sensitivity=sensitivity,
        mood_settings=mood_settings,
        scale=scale,
        channel=ch_melody,
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )

    # [EN] Post-production quantization: accompaniment uses a quarter-note grid;
    # [EN] melody uses its finer expressive grid.
    # [中文] 后处理量化：伴奏使用四分网格；旋律保留更细网格。
    harmony_events = quantize_track_note_events(
        harmony_events,
        ticks_per_beat,
        ch_harmony,
        HARMONY_RANGE[0],
        HARMONY_RANGE[1],
        rhythm_steps_per_beat=harmony_rhythm_steps_per_beat,
    )
    bass_events = quantize_track_note_events(
        bass_events,
        ticks_per_beat,
        ch_bass,
        BASS_RANGE[0],
        BASS_RANGE[1],
        rhythm_steps_per_beat=bass_pad_rhythm_steps_per_beat,
    )
    pad_events = quantize_track_note_events(
        pad_events,
        ticks_per_beat,
        ch_pad,
        PAD_RANGE[0],
        PAD_RANGE[1],
        rhythm_steps_per_beat=bass_pad_rhythm_steps_per_beat,
    )
    melody_events = quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        MELODY_RANGE[0],
        MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )

    def scale_pcs_at_tick(tick: int) -> set:
        _ = tick
        return {int(deg) % 12 for deg in scale}

    # [EN] If a simultaneous melody tone is out of local scale or harshly clashes,
    # [EN] delay it on-grid and shorten as a stepping note.
    # [中文] 若同刻旋律音脱离局部调内或冲突明显，则在网格上后移并缩短为经过音。
    melody_events = resolve_melody_clashes_after_quantization(
        melody_events,
        harmony_events,
        bass_events,
        pad_events,
        ticks_per_beat=ticks_per_beat,
        melody_channel=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick,
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )

    # [EN] Final timing lock after clash-resolution: keep all modules on the same meter grid.
    # [中文] 冲突修正后进行最终对齐：确保全部模块仍在同一拍感网格上。
    harmony_events = quantize_track_note_events(
        harmony_events,
        ticks_per_beat,
        ch_harmony,
        HARMONY_RANGE[0],
        HARMONY_RANGE[1],
        rhythm_steps_per_beat=harmony_rhythm_steps_per_beat,
    )
    bass_events = quantize_track_note_events(
        bass_events,
        ticks_per_beat,
        ch_bass,
        BASS_RANGE[0],
        BASS_RANGE[1],
        rhythm_steps_per_beat=bass_pad_rhythm_steps_per_beat,
    )
    pad_events = quantize_track_note_events(
        pad_events,
        ticks_per_beat,
        ch_pad,
        PAD_RANGE[0],
        PAD_RANGE[1],
        rhythm_steps_per_beat=bass_pad_rhythm_steps_per_beat,
    )
    melody_events = quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        MELODY_RANGE[0],
        MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )

    # [EN] Objective post-metric gate with automatic safe fallback if needed.
    # [中文] 通过客观后处理指标触发自动安全回退。
    harmony_events, bass_events, pad_events, melody_events, _post_metrics, _fallback_used = maybe_apply_safe_post_fallback(
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
        mapping_bias=float(mapping_bias),
    )

    # [EN] Final post check: melody must remain logical with finalized harmony/bass/pad context.
    # [中文] 最终后检：旋律需与最终和声/低音/铺底上下文保持逻辑一致。
    melody_events = resolve_melody_clashes_after_quantization(
        melody_events,
        harmony_events,
        bass_events,
        pad_events,
        ticks_per_beat=ticks_per_beat,
        melody_channel=ch_melody,
        total_ticks=total_ticks,
        scale_pcs_at_tick_fn=scale_pcs_at_tick,
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )
    melody_events = quantize_track_note_events(
        melody_events,
        ticks_per_beat,
        ch_melody,
        MELODY_RANGE[0],
        MELODY_RANGE[1],
        rhythm_steps_per_beat=melody_rhythm_steps_per_beat,
    )

    if synth_output:
        # [EN] Render with the built-in oscillator/additive synthesis pipeline.
        # [中文] 通过内置振荡器/加法合成管线完成渲染输出。
        out_path = output
        if out_path.lower().endswith(".mid"):
            out_path = os.path.splitext(out_path)[0] + ".wav"
        elif not out_path.lower().endswith(".wav"):
            out_path = out_path + ".wav"
        synthesize_to_wav(
            out_path,
            tempo_bpm=tempo_bpm,
            ticks_per_beat=ticks_per_beat,
            sample_rate=int(max(8000, sample_rate)),
            harmony_events=harmony_events,
            bass_events=bass_events,
            melody_events=melody_events,
            pad_events=pad_events,
        )
        return

    if direct_output:
        # [EN] SoundFont pipeline: temporary MIDI -> fluidsynth -> WAV.
        # [中文] SoundFont 路径：临时 MIDI -> fluidsynth -> WAV。
        if not soundfont:
            raise RuntimeError("开启 --direct-output 时必须提供 --soundfont /path/to.sf2")
        out_path = output
        if out_path.lower().endswith(".mid"):
            out_path = os.path.splitext(out_path)[0] + ".wav"
        elif not out_path.lower().endswith(".wav"):
            out_path = out_path + ".wav"

        # [EN] Keep intermediate render artifacts in a temporary directory.
        # [中文] 中间渲染文件写入临时目录，并在流程结束后自动清理。
        with tempfile.TemporaryDirectory(prefix="audio_gen_render_") as td:
            midi_path = os.path.join(td, "render.mid")
            write_temp_midi_for_render(
                midi_path,
                tempo_bpm=tempo_bpm,
                ticks_per_beat=ticks_per_beat,
                tracks=[
                    ("Harmony", harmony_events),
                    ("Bass", bass_events),
                    ("Melody", melody_events),
                    ("Pad", pad_events),
                ],
            )
            render_midi_to_wav_with_fluidsynth(
                midi_path=midi_path,
                wav_path=out_path,
                soundfont_path=soundfont,
                sample_rate=int(max(8000, sample_rate)),
                gain=float(fluidsynth_gain),
            )
        return

    if logic_structure:
        # [EN] Export a Logic-friendly per-track MIDI bundle.
        # [中文] 导出适配 Logic 的分轨 MIDI 文件集合。
        write_logic_bundle(
            output,
            tempo_bpm,
            harmony_events,
            bass_events,
            melody_events,
            pad_events,
            auto_instrument,
        )
        # [EN] Also export a combined reference MIDI for quick verification.
        # [中文] 额外导出合并参考 MIDI，便于快速核对结果。
        combined_path = os.path.join(output, "Combined.mid")
        combined = []
        combined.extend(harmony_events)
        combined.extend(bass_events)
        combined.extend(melody_events)
        combined.extend(pad_events)
        combined_track = render_track(combined)
        combined_track = prepend_meta(
            combined_track,
            [
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)),
                mido.MetaMessage("track_name", name="Combined"),
            ],
        )
        combined_midi = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
        combined_midi.tracks.append(combined_track)
        combined_midi.save(combined_path)
        return

    if single_channel:
        combined = []
        combined.extend(harmony_events)
        combined.extend(bass_events)
        combined.extend(melody_events)
        combined.extend(pad_events)
        combined_track = render_track(combined)
        # [EN] Type-0 MIDI requires a single merged track.
        # [中文] Type-0 MIDI 规范要求单轨，因此此处合并所有事件。
        combined_track.insert(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
        midi.tracks = [combined_track]
    else:
        harmony_track = render_track(harmony_events)
        bass_track = render_track(bass_events)
        melody_track = render_track(melody_events)
        pad_track = render_track(pad_events)

        if logic_pro:
            harmony_track = prepend_meta(
                harmony_track,
                [
                    mido.MetaMessage("track_name", name="Harmony"),
                    mido.MetaMessage("instrument_name", name="Harmony"),
                ],
            )
            bass_track = prepend_meta(
                bass_track,
                [
                    mido.MetaMessage("track_name", name="Bass"),
                    mido.MetaMessage("instrument_name", name="Bass"),
                ],
            )
            melody_track = prepend_meta(
                melody_track,
                [
                    mido.MetaMessage("track_name", name="Melody"),
                    mido.MetaMessage("instrument_name", name="Melody"),
                ],
            )
            pad_track = prepend_meta(
                pad_track,
                [
                    mido.MetaMessage("track_name", name="Pad"),
                    mido.MetaMessage("instrument_name", name="Pad"),
                ],
            )

        midi.tracks.append(harmony_track)
        midi.tracks.append(bass_track)
        midi.tracks.append(melody_track)
        midi.tracks.append(pad_track)

    midi.save(output)


# ============================================================================
# [EN] Orchestration
# [中文] 流水线调度
# ============================================================================

def is_video(path: str) -> bool:
    """Purpose / 目的
    EN: Determine whether an input path should be processed as video.
    中文：判断输入路径是否按视频模式处理。"""
    _, ext = os.path.splitext(path.lower())
    return ext in VIDEO_EXTS


def discover_batch_inputs(input_dir: str) -> List[Tuple[int, str]]:
    """EN: Find input-<n>.* files in a directory and sort by numeric id.
    中文：在目录中查找 input-<n>.* 文件并按编号排序。"""
    items: List[Tuple[int, str]] = []
    for name in os.listdir(input_dir):
        m = INPUT_BATCH_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        items.append((idx, os.path.join(input_dir, name)))
    items.sort(key=lambda x: x[0])
    return items


def run_single_pipeline(
    args: argparse.Namespace,
    *,
    input_path: str,
    output_path: str,
    progress_desc: Optional[str] = None,
) -> None:
    """EN: Execute one full input->output conversion job.
    中文：执行一次完整的输入到输出转换任务。"""
    np.random.seed(args.seed)

    if not is_video(input_path) and not args.line_scan:
        raise SystemExit("Warning：Trying line scan? add --line-scan")

    segment_infos: List[SegmentInfo] = []
    global_pixels: Optional[np.ndarray] = None
    accent: List[float] = []
    fps = 30.0
    segment_ranges: List[Tuple[int, int]] = []
    signal_s: List[float] = []
    #已弃用
    # [EN] Step 1: Load timeline frames (video) or strips (line-scan debug mode).
    # [中文] 步骤1：加载时间帧（视频）或条带（行扫描调试模式）。
    if is_video(input_path):
        # [EN] Video mode uses the decoded frame sequence as the primary timeline.
        # [中文] 视频模式以解码帧序列作为主时间线。
        frames, fps = load_video_frames(
            input_path,
            args.max_size,
            args.frame_step,
            args.max_frames,
            progress_desc=progress_desc,
        )
        if not frames:
            raise RuntimeError("未读取到任何视频帧。")
        # [EN] S(t) is the master signal driving segmentation, rhythmic density, and accents.
        # [中文] S(t) 是驱动分段、密度与重音的主信号。
        signal_s, _, _ = build_signal_s(frames, args.smooth)
        accent = accent_strength(signal_s)
        segment_ranges = segment_by_signal(signal_s, args.frames_per_segment)
        if not segment_ranges:
            segment_ranges = [(0, len(frames))]
        # [EN] Sample global pixels across frames to obtain stable global descriptors.
        # [中文] 跨帧采样全局像素，以提升全局描述符稳定性。
        global_pixels = sample_pixels(np.concatenate([f.reshape(-1, 3) for f in frames], axis=0), args.sample, args.seed)

    else:
        #已弃用
        # [EN] Line-scan mode converts image strips into pseudo-time frames.
        # [中文] 行扫描模式将图像条带转换为伪时间帧。
        img = load_image(input_path, args.max_size)
        hsv = rgb_to_hsv_array(img)
        _segments, strips = compute_line_scan_segments(hsv, args.slices)
        frames = strips
        fps = 1.0
        signal_s, _, _ = build_signal_s(frames, args.smooth)
        accent = accent_strength(signal_s)
        segment_ranges = segment_by_signal(signal_s, max(1, args.slices // 4))
        global_pixels = sample_pixels(hsv.reshape(-1, 3), args.sample, args.seed)

    # [EN] Step 2: Compute global descriptors for tempo estimation and deterministic seed basis.
    # [中文] 步骤2：计算全局描述符，用于速度估计与确定性种子。
    if global_pixels is None:
        raise RuntimeError("未能计算全局像素特征。")
    if args.method == "kmeans":
        global_clusters = compute_clusters_kmeans(global_pixels, args.k, args.seed)
    else:
        global_clusters = compute_clusters_histogram(global_pixels, args.bins, args.k)

    entropy_norm = luminance_entropy(global_pixels)
    r_opt_norm = optical_span_norm(global_clusters)
    auto_bpm = select_bpm(entropy_norm, r_opt_norm)
    # [EN] Global S(t) statistics govern rhythmic subdivision tendency.
    # [中文] S(t) 全局统计量决定节奏细分倾向。
    mu = float(np.mean(signal_s)) if signal_s else 0.0
    sigma = float(np.var(signal_s)) if signal_s else 0.0
    # [EN] Final seed = user seed + S(t) fingerprint (reproducible for the same video).
    # [中文] 最终种子 = 用户种子 + S(t) 指纹（同一视频可复现）。
    signal_hash = int(np.sum(np.round(np.asarray(signal_s) * 1000.0))) & 0xFFFFFFFF
    seed = int((args.seed * 1315423911 + signal_hash + len(signal_s) * 2654435761) & 0xFFFFFFFF)

    # [EN] Step 3: Build segment features for harmony, register, and melodic control.
    # [中文] 步骤3：构建分段特征，用于和声、音区与旋律控制。
    feature_list: List[SegmentFeatures] = []
    ranges_filtered: List[Tuple[int, int]] = []
    for start, end in segment_ranges:
        block = frames[start:end]
        if not block:
            continue
        merged = np.concatenate([f.reshape(-1, 3) for f in block], axis=0)
        pixels = sample_pixels(merged, args.sample, args.seed + start)
        if args.method == "kmeans":
            clusters = compute_clusters_kmeans(pixels, args.k, args.seed + start)
        else:
            clusters = compute_clusters_histogram(pixels, args.bins, args.k)
        features = build_segment_features(clusters)
        feature_list.append(features)
        ranges_filtered.append((start, end))
    if not feature_list:
        raise RuntimeError("未能生成任何分段特征。")

    feature_list = smooth_segments(feature_list, max(1, args.smooth))
    for (start, end), features in zip(ranges_filtered, feature_list):
        segment_infos.append(SegmentInfo(features=features, start=start, end=end))

    # [EN] Step 4: Determine musical context (scale and tempo).
    # [中文] 步骤4：确定音乐上下文（调式与速度）。
    mood_settings = MOODS.get(args.mood, {}) if args.mood else {}
    scale_override = args.scale if args.scale else mood_settings.get("scale")
    scale_name = choose_scale([s.features for s in segment_infos], scale_override)

    global_valence, global_arousal = global_affect(segment_infos)
    affective_bpm = select_bpm_affective(global_arousal)
    auto_bpm = clamp_int(int(round(0.35 * auto_bpm + 0.65 * affective_bpm)), 40, 200)

    tempo_bpm = args.tempo if args.tempo is not None else auto_bpm
    if args.tempo is None and mood_settings.get("tempo") is not None:
        tempo_bpm = int(round(tempo_bpm * (float(mood_settings["tempo"]) / 120.0)))
    tempo_bpm = clamp_int(tempo_bpm, 40, 200)

    out_path = output_path
    if args.logic_structure:
        if out_path.lower().endswith(".mid"):
            out_path = os.path.splitext(out_path)[0] + "_logic"
        os.makedirs(out_path, exist_ok=True)

    # [EN] Step 5: Generate musical events and export or render the selected output format.
    # [中文] 步骤5：生成音乐事件并导出或渲染所选输出格式。
    generate_midi(
        segments=segment_infos,
        accent=accent,
        fps=fps,
        total_frames=len(frames),
        mu=mu,
        sigma=sigma,
        seed=seed,
        output=out_path,
        tempo_bpm=tempo_bpm,
        bars_per_segment=args.bars_per_segment,
        scale_name=scale_name,
        single_channel=args.single_channel,
        logic_pro=args.logic_pro,
        auto_instrument=args.auto_instrument,
        sensitivity=args.sensitivity,
        mapping_bias=float(getattr(args, "mapping_bias", 0.5)),
        mood=args.mood,
        logic_structure=args.logic_structure,
        direct_output=bool(args.direct_output),
        synth_output=bool(args.synth_output),
        sample_rate=int(args.sample_rate),
        soundfont=args.soundfont,
        fluidsynth_gain=float(args.fluidsynth_gain),
    )


def main() -> None:
    """Purpose / 目的
    EN: Orchestrate the full video-to-music workflow end to end.
    中文：调度完整的视频到音乐流水线。

    Workflow / 流程
    EN:
    1) parse CLI args and load timeline frames
    2) compute S(t), non-uniform segments, and accent peaks
    3) compute global/segment visual descriptors
    4) derive BPM, scale, and deterministic seed
    5) generate events and export/render output
    中文：
    1）解析参数并加载时间帧
    2）计算 S(t)、非均匀分段与重音峰值
    3）计算全局与分段视觉描述符
    4）推导 BPM、调式与确定性种子
    5）生成事件并导出/渲染结果
    """
    args = parse_args()

    if args.batch:
        input_dir = os.path.abspath(args.input)
        output_dir = os.path.abspath(args.output)
        if not os.path.isdir(input_dir):
            raise SystemExit(f"Batch mode requires an input directory: {input_dir}")
        if args.line_scan:
            raise SystemExit("Batch mode does not support --line-scan.")
        if args.logic_structure:
            raise SystemExit("Batch mode expects audio conversion; disable --logic-structure.")
        if not args.synth_output and not args.direct_output:
            # [EN] Batch conversion targets WAV files by default.
            # [中文] 批处理默认目标是 WAV 音频。
            args.synth_output = True

        jobs = discover_batch_inputs(input_dir)
        if not jobs:
            raise SystemExit("No matching input files found (expected input-<n>.mp4 style names).")
        os.makedirs(output_dir, exist_ok=True)

        failed: List[Tuple[str, str]] = []
        total = len(jobs)
        for idx, (num, in_path) in enumerate(jobs, start=1):
            out_path = os.path.join(output_dir, f"output-{num}.wav")
            tag = f"[{idx}/{total}] {os.path.basename(in_path)}"
            print(f"{tag} -> {os.path.basename(out_path)}", flush=True)
            try:
                run_single_pipeline(args, input_path=in_path, output_path=out_path, progress_desc=tag)
            except Exception as exc:
                failed.append((in_path, str(exc)))
                print(f"[FAILED] {os.path.basename(in_path)}: {exc}")

        if failed:
            msg = "; ".join([f"{os.path.basename(p)} ({e})" for p, e in failed[:5]])
            raise SystemExit(f"Batch finished with {len(failed)} failures: {msg}")
        print(f"Batch completed: {total} files -> {output_dir}")
        return

    run_single_pipeline(args, input_path=args.input, output_path=args.output, progress_desc=os.path.basename(args.input))


if __name__ == "__main__":
    main()

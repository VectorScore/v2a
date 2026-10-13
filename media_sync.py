from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Optional

import imageio.v2 as imageio

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


def resolve_ffmpeg_exe(cli_value: Optional[str] = None) -> str:
    candidates = []
    if cli_value:
        candidates.append(str(cli_value))
    env_value = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_value:
        candidates.append(env_value)
    if imageio_ffmpeg is not None:
        try:
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        candidates.append(sys_ffmpeg)
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.exists():
            return str(p)
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
    raise RuntimeError("ffmpeg binary not found.")


def probe_video_duration_seconds(path_text: str) -> float:
    reader = None
    try:
        reader = imageio.get_reader(str(path_text))
        meta = reader.get_meta_data()
        duration = float(meta.get("duration", 0.0))
        if math.isfinite(duration) and duration > 0.0:
            return duration
    finally:
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
    raise RuntimeError(f"Unable to determine video duration: {path_text}")


def probe_wav_duration_seconds(path_text: str) -> float:
    with wave.open(str(path_text), "rb") as wav_file:
        frames = int(wav_file.getnframes())
        rate = int(wav_file.getframerate())
    if rate <= 0:
        raise RuntimeError(f"Invalid WAV sample rate: {path_text}")
    return float(frames) / float(rate)


def probe_media_duration_seconds(path_text: str) -> float:
    path = Path(path_text)
    if path.suffix.lower() == ".wav":
        return probe_wav_duration_seconds(str(path))
    return probe_video_duration_seconds(str(path))


def _build_atempo_chain(total_factor: float) -> str:
    factor = float(total_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise RuntimeError(f"Invalid atempo factor: {factor}")
    filters = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    filters.append(f"atempo={factor:.8f}")
    return ",".join(filters)


def sync_wav_to_video_duration(
    wav_path: str,
    video_path: str,
    *,
    ffmpeg_exe: Optional[str] = None,
    tolerance_sec: float = 0.05,
) -> Dict[str, float | bool | str]:
    ffmpeg_bin = resolve_ffmpeg_exe(ffmpeg_exe)
    wav_file = Path(wav_path).expanduser().resolve()
    video_file = Path(video_path).expanduser().resolve()
    target_duration = probe_video_duration_seconds(str(video_file))
    audio_before = probe_wav_duration_seconds(str(wav_file))
    delta_before = abs(audio_before - target_duration)
    if delta_before <= float(max(0.0, tolerance_sec)):
        return {
            "ffmpeg_exe": ffmpeg_bin,
            "modified": False,
            "video_duration": target_duration,
            "audio_duration_before": audio_before,
            "audio_duration_after": audio_before,
            "delta_before": delta_before,
            "delta_after": delta_before,
            "tempo_factor": 1.0,
        }

    tempo_factor = float(audio_before / max(1e-9, target_duration))
    filter_expr = (
        f"{_build_atempo_chain(tempo_factor)},"
        f"apad=whole_dur={target_duration:.6f},"
        f"atrim=0:{target_duration:.6f}"
    )

    with tempfile.NamedTemporaryFile(
        prefix=f"{wav_file.stem}_sync_",
        suffix=".wav",
        dir=str(wav_file.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_file),
        "-vn",
        "-filter:a",
        filter_expr,
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg tempo-sync failed: {err or 'unknown error'}")
        tmp_path.replace(wav_file)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    audio_after = probe_wav_duration_seconds(str(wav_file))
    delta_after = abs(audio_after - target_duration)
    return {
        "ffmpeg_exe": ffmpeg_bin,
        "modified": True,
        "video_duration": target_duration,
        "audio_duration_before": audio_before,
        "audio_duration_after": audio_after,
        "delta_before": delta_before,
        "delta_after": delta_after,
        "tempo_factor": tempo_factor,
    }


def mux_video_with_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    ffmpeg_exe: Optional[str] = None,
) -> Dict[str, float | str]:
    ffmpeg_bin = resolve_ffmpeg_exe(ffmpeg_exe)
    video_file = Path(video_path).expanduser().resolve()
    audio_file = Path(audio_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_file),
        "-i",
        str(audio_file),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg mux failed: {err or 'unknown error'}")

    return {
        "ffmpeg_exe": ffmpeg_bin,
        "video_duration": probe_video_duration_seconds(str(video_file)),
        "audio_duration": probe_wav_duration_seconds(str(audio_file)),
        "output_duration": probe_video_duration_seconds(str(output_file)),
        "output_path": str(output_file),
    }

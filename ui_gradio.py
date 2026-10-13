#!/usr/bin/env python3
"""Single-page Gradio UI for v2a-nyq (video -> Nyquist synthesized WAV)."""

from __future__ import annotations

import html
import json
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import gradio as gr

from media_sync import mux_video_with_audio, probe_video_duration_seconds, probe_wav_duration_seconds

# Avoid localhost requests being routed through system HTTP proxies.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "convert-tool.py"
DEFAULT_OUT_DIR = ROOT / "output"
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALES = ["major", "minor", "dorian", "mixolydian", "pentatonic"]
MOODS = ["none", "cheerful", "epic"]

TASK_HEADERS = ["Task ID", "Status", "Estimated finish", "Started", "Ended", "Input", "Output"]
TASKS: List[Dict[str, Any]] = []
TASK_LOCK = threading.Lock()
NEXT_TASK_ID = 1
TASK_PROCESSES: Dict[int, subprocess.Popen[str]] = {}

TOOLTIPS_EN: Dict[str, str] = {
    "k": "Dominant Color Clusters (k): how many palette clusters are preserved during numeric feature extraction.",
    "method": "Color Feature Method\n\nkmeans:\n- Learns dominant HSV clusters.\n- Better palette structure retention.\n- Slightly slower.\n\nhist:\n- Uses direct HSV statistics.\n- Faster and simpler.\n- Coarser mapping detail.",
    "frame_step": "Frame Sampling Step: decode every Nth frame. Higher is faster but loses temporal detail.",
    "frames_per_segment": "Segment Length (Frames): sampled frames grouped into one Nyquist composition segment.",
    "smooth": "Segment Smoothing Window\n\nAfter feature extraction, each segment is averaged with neighboring segments in a local window.\n- Averages: saturation/value means, variances, and motion.\n- Hue uses circular averaging (handles wrap-around at 0/1 correctly).\n\nThis affects all musical generation (harmony, bass, melody, dynamics), not only melody.\n\nValue guide:\n- 1: no smoothing (most reactive, more jitter).\n- 3: balanced default (uses immediate neighbors).\n- 5+: smoother/more legato behavior, but slower response to rapid visual changes.",
    "max_frames": "Maximum Frames: upper bound of decoded sampled frames to control runtime.",
    "sensitivity": "Visual Sensitivity: scales how strongly visual feature changes affect Nyquist note behavior.",
    "tempo_auto": "Auto Tempo: Nyquist computes BPM from extracted motion/contrast features.",
    "scale_auto": "Auto Scale/Mode: Nyquist chooses scale from extracted hue/spread/brightness dynamics.",
    "scale": "Scale/Mode: manual override when Auto Scale/Mode is disabled.",
    "mood": "Mood Preset: style bias for Nyquist composition model (none/cheerful/epic).",
    "mapping_bias": "Safety Correction -- Pure Visual Mapping: left is more conservative; right follows visual dynamics more literally.",
    "nyquist_bin": "Optional Nyquist binary path. Leave empty to use `ny`/`nyquist` from PATH.",
    "nyquist_timeout": "Nyquist Timeout (sec): maximum allowed synthesis time before aborting. Increase for longer/high-detail videos.",
}

UI_CSS = """
.param-info-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  margin-left: 6px;
  border: 1px solid #8392a6;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  color: #1f3a5f;
  background: #eef3fa;
  cursor: help;
  user-select: none;
  vertical-align: middle;
}
.param-info-popup {
  position: absolute;
  z-index: 99999;
  max-width: 360px;
  padding: 10px 12px;
  border: 1px solid #70829b;
  border-radius: 10px;
  background: #ffffff;
  color: #162a43;
  box-shadow: 0 10px 28px rgba(21, 38, 63, 0.22);
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-line;
  display: none;
  pointer-events: none;
}
.upload-banner {
  margin: 6px 0 10px 0;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #7fb88c;
  background: #ecf8ef;
  color: #1f4f2c;
  font-size: 13px;
}
#video_preview {
  max-width: 420px;
}
#video_preview .wrap,
#video_preview .empty,
#video_preview video,
#video_preview iframe {
  width: 100% !important;
  height: 236px !important;
  max-height: 236px !important;
  min-height: 236px !important;
}
#video_preview video,
#video_preview iframe {
  object-fit: contain;
  background: #000;
  border-radius: 10px;
}
#task_queue {
  margin: 4px 0;
  padding: 8px;
  border: 1px solid #d6dee8;
  border-radius: 12px;
  overflow-x: auto;
}
#task_queue .label { margin-bottom: 4px; font-size: 12px; }
#task_queue table { width: 100%; min-width: 0; table-layout: fixed; }
#task_queue th, #task_queue td {
  padding: 5px 7px;
  font-size: 12px;
  line-height: 1.25;
  white-space: normal !important;
}
#task_queue th { overflow-wrap: normal !important; word-break: normal !important; }
#task_queue td { overflow-wrap: anywhere !important; word-break: break-word; }
#task_queue th [role="button"] { white-space: normal !important; overflow-wrap: normal !important; text-align: left; }
.gradio-container { --layout-gap: 10px; --block-padding: 8px 10px; }
#task_queue th .cell-wrap { min-height: 0; }
#task_queue th .sort-button { padding: 2px; }
.top-input-row { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); align-items: stretch; gap: 10px; }
#destination_controls { min-width: 0 !important; gap: 8px; }
#input_video, #gen_btn { height: auto !important; min-height: 0 !important; width: 100%; }
#input_video .wrap, #input_video > button { min-height: 0 !important; height: 100% !important; }
#fast_preset:disabled { cursor: not-allowed !important; pointer-events: auto !important; }
#task_queue th [role="button"] { font-size: 12px; line-height: 1.25; padding: 2px 0; min-height: 0; }
#video_preview:fullscreen, #video_preview :fullscreen { max-width: none !important; width: 100vw !important; height: 100vh !important; max-height: none !important; }
#video_preview :fullscreen video { height: 100% !important; max-height: none !important; }
@media (max-width: 600px) { .top-input-row { grid-template-columns: 1fr; } #input_video, #gen_btn { min-height: 90px !important; } }
.site-header { align-items: flex-start; }
.site-header .prose { margin-bottom: 0; }
#about_us { display: flex; justify-content: flex-end; position: relative; z-index: 20; }
#about_us button {
  border: 1px solid #a7b5c7;
  border-radius: 9px;
  background: #f4f7fb;
  color: #24364d;
  padding: 9px 14px;
  font-weight: 600;
  white-space: nowrap;
}
#about_us button:hover { background: #e9f0f8; }
#about_panel {
  position: fixed;
  inset: auto 16px auto auto;
  top: 104px;
  width: min(410px, calc(100vw - 32px));
  padding: 18px 20px;
  border: 1px solid #cbd6e3;
  border-radius: 12px;
  background: #fff;
  color: #24364d;
  box-shadow: 0 12px 30px rgba(27, 45, 70, .18);
  line-height: 1.55;
}
#about_panel p { margin: 0 0 12px; }
#about_panel p:last-child { margin-bottom: 0; }
"""


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _task_rows() -> List[List[str]]:
    with TASK_LOCK:
        rows: List[List[str]] = []
        for t in TASKS:
            rows.append(
                [
                    str(t["id"]),
                    str(t["status"]),
                    _estimated_finish(t),
                    str(t["started"]),
                    str(t.get("ended", "")),
                    str(t["input"]),
                    str(t["output"]),
                ]
            )
        return rows


def _estimated_finish(task: Dict[str, Any]) -> str:
    if task["status"] == "Queued":
        return "After current job"
    if task["status"] != "Running":
        return "—"
    elapsed = max(0.0, time.time() - float(task["run_started_at"]))
    baseline = float(task.get("estimate_seconds", 180.0))
    if elapsed > baseline * 2:
        return "Delayed; ETA unknown"
    # This is a planning estimate, not a deadline; long decoding or synthesis
    # can move it later. Never show an ETA that has already passed.
    total = max(baseline, elapsed * 1.25)
    return time.strftime("~%H:%M (rough)", time.localtime(time.time() + total - elapsed))


def _task_create(input_path: str, output_path: str, max_frames: int, frame_step: int, method: str) -> int:
    global NEXT_TASK_ID
    with TASK_LOCK:
        task_id = NEXT_TASK_ID
        NEXT_TASK_ID += 1
        TASKS.append(
            {
                "id": task_id,
                "status": "Queued",
                "started": _now_str(),
                "ended": "",
                "input": Path(input_path).name,
                "output": Path(output_path).name,
                "run_started_at": 0.0,
                "cancel_requested": False,
                "estimate_seconds": 60 + min(int(max_frames), 1200) * (0.7 if method == "hist" else 1.1),
            }
        )
    return task_id


def _task_update(task_id: int, **fields: Any) -> None:
    with TASK_LOCK:
        for t in TASKS:
            if t["id"] == task_id:
                t.update(fields)
                break


def _cancel_task(task_id: float) -> tuple[str, List[List[str]]]:
    try:
        selected_id = int(task_id)
    except (TypeError, ValueError):
        return "Enter a task ID to cancel.", _task_rows()
    with TASK_LOCK:
        task = next((t for t in TASKS if t["id"] == selected_id), None)
        if task is None:
            message = f"Task {selected_id} was not found."
        elif task["status"] not in ("Queued", "Running"):
            message = f"Task {selected_id} has already finished."
        else:
            task["cancel_requested"] = True
            task["status"] = "Cancelling"
            proc = TASK_PROCESSES.get(selected_id)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            message = f"Cancellation requested for task {selected_id}."
    return message, _task_rows()


def _toggle_by_auto(is_auto: bool) -> gr.update:
    return gr.update(interactive=not bool(is_auto))


def _apply_fast_preset() -> tuple[int, str, int, int, int, int, float, float, int]:
    # Fast but mapping-preserving preset:
    # - keep finer temporal detail than the previous fast profile
    # - reduce over-smoothing that caused repetitive harmonic cells
    return (5, "hist", 2, 6, 1, 180, 1.3, 0.78, 120)


def _sanitize_output_dir(path_text: str) -> str:
    out_dir = Path(path_text or str(DEFAULT_OUT_DIR)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _flatten_picker_paths(value: Any) -> List[str]:
    paths: List[str] = []
    if value is None:
        return paths
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        p = value.get("path")
        if isinstance(p, str) and p:
            return [p]
        return paths
    if hasattr(value, "name"):
        p = getattr(value, "name")
        if isinstance(p, str) and p:
            return [p]
        return paths
    if isinstance(value, (list, tuple)):
        for item in value:
            paths.extend(_flatten_picker_paths(item))
    return paths


def _resolve_output_dir_from_picker(picked: Any, current_path: str) -> str:
    fallback = _sanitize_output_dir(current_path)
    raw_paths = [p for p in _flatten_picker_paths(picked) if isinstance(p, str) and p.strip()]
    if not raw_paths:
        return fallback
    abs_paths: List[str] = []
    for p in raw_paths:
        try:
            abs_paths.append(str(Path(p).expanduser().resolve()))
        except Exception:
            continue
    if not abs_paths:
        return fallback
    if len(abs_paths) == 1:
        one = Path(abs_paths[0])
        chosen = str(one if one.is_dir() else one.parent)
        return _sanitize_output_dir(chosen)
    try:
        common = os.path.commonpath(abs_paths)
    except Exception:
        common = str(Path(abs_paths[0]).parent)
    return _sanitize_output_dir(common)


def _build_output_path(input_path: str, output_dir: str) -> str:
    stem = Path(input_path).stem or "output"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(output_dir) / f"{stem}_{stamp}.wav")


def _build_video_output_path(audio_output_path: str) -> str:
    return str(Path(audio_output_path).with_suffix(".mp4"))


def _build_midi_zip_path(audio_output_path: str) -> str:
    audio_path = Path(audio_output_path)
    return str(audio_path.with_name(f"{audio_path.stem}_midi_stems.zip"))


def _on_input_uploaded(input_path: Optional[str]) -> gr.update:
    if not input_path:
        return gr.update(value="", visible=False)
    p = Path(input_path)
    name = html.escape(p.name)
    size_mb = 0.0
    try:
        size_mb = p.stat().st_size / (1024.0 * 1024.0)
    except Exception:
        size_mb = 0.0
    return gr.update(
        visible=True,
        value=f"<div class='upload-banner'><b>Upload complete.</b> File: <b>{name}</b> ({size_mb:.1f} MB). You can now generate audio.</div>",
    )


def _hide_upload_banner() -> gr.update:
    return gr.update(value="", visible=False)


def _build_command(
    *,
    input_path: str,
    output_path: str,
    nyquist_bin: str,
    k: int,
    method: str,
    frame_step: int,
    frames_per_segment: int,
    smooth: int,
    tempo_auto: bool,
    tempo_value: int,
    scale_auto: bool,
    scale_value: str,
    mood: str,
    sensitivity: float,
    max_frames: int,
    mapping_bias: float,
    nyquist_timeout: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        input_path,
        output_path,
        "--synth-output",
        "--k",
        str(int(k)),
        "--method",
        str(method),
        "--frame-step",
        str(int(frame_step)),
        "--frames-per-segment",
        str(int(frames_per_segment)),
        "--smooth",
        str(int(smooth)),
        "--max-frames",
        str(int(max_frames)),
        "--sensitivity",
        f"{float(sensitivity):.3f}",
        "--mapping-bias",
        f"{float(mapping_bias):.3f}",
        "--nyquist-timeout",
        str(int(nyquist_timeout)),
    ]
    if nyquist_bin.strip():
        cmd.extend(["--nyquist-bin", nyquist_bin.strip()])
    if not bool(tempo_auto):
        cmd.extend(["--tempo", str(int(tempo_value))])
    if not bool(scale_auto):
        cmd.extend(["--scale", str(scale_value)])
    if mood and mood != "none":
        cmd.extend(["--mood", str(mood)])
    return cmd


def _run_generator(
    input_path: str,
    output_dir: str,
    nyquist_bin: str,
    k: int,
    method: str,
    frame_step: int,
    frames_per_segment: int,
    smooth: int,
    tempo_auto: bool,
    tempo_value: int,
    scale_auto: bool,
    scale_value: str,
    mood: str,
    sensitivity: float,
    max_frames: int,
    mapping_bias: float,
    nyquist_timeout: int,
    state: Optional[Dict[str, Any]],
):
    logs: List[str] = []
    state = state or {}

    if not input_path:
        logs.append("No input video selected.")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    if not SCRIPT_PATH.exists():
        logs.append(f"Script not found: {SCRIPT_PATH}")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    out_dir = _sanitize_output_dir(output_dir)
    output_path = _build_output_path(input_path, out_dir)
    bias = float(max(0.0, min(1.0, mapping_bias)))

    cmd = _build_command(
        input_path=input_path,
        output_path=output_path,
        nyquist_bin=nyquist_bin,
        k=k,
        method=method,
        frame_step=frame_step,
        frames_per_segment=frames_per_segment,
        smooth=smooth,
        tempo_auto=tempo_auto,
        tempo_value=tempo_value,
        scale_auto=scale_auto,
        scale_value=scale_value,
        mood=mood,
        sensitivity=sensitivity,
        max_frames=max_frames,
        mapping_bias=bias,
        nyquist_timeout=nyquist_timeout,
    )

    task_id = _task_create(input_path, output_path, max_frames, frame_step, method)
    logs.append("[Task] Queued.")
    logs.append(f"[Command] {' '.join(shlex.quote(part) for part in cmd)}")
    yield None, None, None, None, None, "\n".join(logs), _task_rows(), state

    cancelled_before_start = False
    with TASK_LOCK:
        task = next(t for t in TASKS if t["id"] == task_id)
        if task["cancel_requested"]:
            task.update(status="Cancelled", ended=_now_str())
            cancelled_before_start = True
        else:
            task.update(status="Running", run_started_at=time.time())
    if cancelled_before_start:
        logs.append("[Cancelled] Task was cancelled before it started.")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return
    q: Queue[str] = Queue()
    result: Dict[str, Any] = {}

    def worker() -> None:
        proc: Optional[subprocess.Popen[str]] = None
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            with TASK_LOCK:
                TASK_PROCESSES[task_id] = proc
                task = next(t for t in TASKS if t["id"] == task_id)
                if task["cancel_requested"]:
                    os.killpg(proc.pid, signal.SIGTERM)
            assert proc.stdout is not None
            for raw in proc.stdout:
                text = raw.rstrip("\n").replace("\r", "").strip()
                if text:
                    q.put(text)
            result["returncode"] = int(proc.wait())
        except Exception as exc:  # pragma: no cover
            result["error"] = str(exc)
        finally:
            with TASK_LOCK:
                TASK_PROCESSES.pop(task_id, None)
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            q.put("__DONE__")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    done = False
    while not done:
        updated = False
        try:
            while True:
                msg = q.get_nowait()
                if msg == "__DONE__":
                    done = True
                    break
                logs.append(msg)
                if len(logs) > 600:
                    logs = logs[-600:]
                updated = True
        except Empty:
            pass

        if updated:
            yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        if not done:
            time.sleep(0.2)

    with TASK_LOCK:
        cancelled = bool(next(t for t in TASKS if t["id"] == task_id)["cancel_requested"])
    if cancelled:
        _task_update(task_id, status="Cancelled", ended=_now_str())
        logs.append("[Cancelled] Conversion stopped by user.")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    if "error" in result:
        err = str(result["error"])
        _task_update(task_id, status="Failed", ended=_now_str())
        logs.append(f"[Error] {err}")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    returncode = int(result.get("returncode", 1))
    if returncode != 0:
        _task_update(task_id, status="Failed", ended=_now_str())
        logs.append(f"[Failed] Process exit code: {returncode}")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    if not Path(output_path).exists():
        _task_update(task_id, status="Failed", ended=_now_str())
        logs.append(f"[Failed] Output file not found: {output_path}")
        yield None, None, None, None, None, "\n".join(logs), _task_rows(), state
        return

    video_duration = 0.0
    audio_duration = 0.0
    video_output_path = _build_video_output_path(output_path)
    midi_zip_path = _build_midi_zip_path(output_path)
    midi_zip_file = midi_zip_path if Path(midi_zip_path).exists() else None
    if midi_zip_file:
        logs.append(f"[MIDI] ZIP ready: {midi_zip_file}")
    else:
        logs.append(f"[Warning] MIDI ZIP not found: {midi_zip_path}")
    try:
        video_duration = probe_video_duration_seconds(input_path)
        audio_duration = probe_wav_duration_seconds(output_path)
        audio_delta = abs(audio_duration - video_duration)
        logs.append(
            f"[Duration] video={video_duration:.3f}s audio={audio_duration:.3f}s diff={audio_delta:.3f}s"
        )
        if audio_delta > 0.05:
            raise RuntimeError(
                f"Generated WAV duration mismatch after sync: video={video_duration:.3f}s "
                f"audio={audio_duration:.3f}s diff={audio_delta:.3f}s"
            )
        mux_info = mux_video_with_audio(
            video_path=input_path,
            audio_path=output_path,
            output_path=video_output_path,
        )
        mux_duration = float(mux_info["output_duration"])
        mux_delta = abs(mux_duration - video_duration)
        logs.append("[Mux] Original video audio removed and replaced with generated audio.")
        logs.append(
            f"[Mux] MP4 ready: {video_output_path} | video={video_duration:.3f}s "
            f"output={mux_duration:.3f}s diff={mux_delta:.3f}s"
        )
        if mux_delta > 0.05:
            raise RuntimeError(
                f"Synthesized MP4 duration mismatch: video={video_duration:.3f}s "
                f"output={mux_duration:.3f}s diff={mux_delta:.3f}s"
            )
    except Exception as exc:
        _task_update(task_id, status="Partial", ended=_now_str())
        logs.append(f"[Partial] WAV created but synchronized MP4 export failed: {exc}")
        new_state = {
            "output_path": output_path,
            "input_path": input_path,
            "mapping_bias": bias,
            "video_output_path": None,
            "midi_zip_path": midi_zip_file,
        }
        yield None, output_path, output_path, None, midi_zip_file, "\n".join(logs), _task_rows(), new_state
        return

    _task_update(task_id, status="Done", ended=_now_str())
    logs.append("[Done] Generation completed.")
    new_state = {
        "output_path": output_path,
        "input_path": input_path,
        "mapping_bias": bias,
        "video_output_path": video_output_path,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "midi_zip_path": midi_zip_file,
    }
    yield video_output_path, output_path, output_path, video_output_path, midi_zip_file, "\n".join(logs), _task_rows(), new_state


def _tooltip_js() -> str:
    payload = json.dumps(TOOLTIPS_EN, ensure_ascii=False)
    return (
        "() => {"
        f"const tips = {payload};"
        "const dropdownIds = new Set(['method', 'scale', 'mood']);"
        "const ensurePopup = () => {"
        "  let popup = document.getElementById('param-info-popup');"
        "  if (!popup) {"
        "    popup = document.createElement('div');"
        "    popup.id = 'param-info-popup';"
        "    popup.className = 'param-info-popup';"
        "    document.body.appendChild(popup);"
        "  }"
        "  return popup;"
        "};"
        "const placePopup = (popup, icon) => {"
        "  const rect = icon.getBoundingClientRect();"
        "  popup.style.left = `${window.scrollX + rect.right + 8}px`;"
        "  popup.style.top = `${window.scrollY + rect.top - 2}px`;"
        "};"
        "const showPopup = (icon, text) => {"
        "  const popup = ensurePopup();"
        "  popup.textContent = text;"
        "  placePopup(popup, icon);"
        "  popup.style.display = 'block';"
        "};"
        "const hidePopup = () => {"
        "  const popup = document.getElementById('param-info-popup');"
        "  if (popup) popup.style.display = 'none';"
        "};"
        "const addIcon = (labelNode, tip) => {"
        "  if (!labelNode || labelNode.querySelector('.param-info-icon')) return;"
        "  const icon = document.createElement('span');"
        "  icon.className = 'param-info-icon';"
        "  icon.textContent = 'i';"
        "  icon.setAttribute('aria-label', 'Parameter info');"
        "  icon.addEventListener('mouseenter', () => showPopup(icon, tip));"
        "  icon.addEventListener('mousemove', () => showPopup(icon, tip));"
        "  icon.addEventListener('mouseleave', hidePopup);"
        "  labelNode.appendChild(icon);"
        "};"
        "const pickLabelNode = (root) => {"
        "  if (!root) return null;"
        "  const selectors = ['.label-wrap', '.label', '.wrap > label', 'label', '.block-title', '[data-testid=\"block-label\"]'];"
        "  for (const sel of selectors) {"
        "    const node = root.querySelector(sel);"
        "    if (node) return node;"
        "  }"
        "  return null;"
        "};"
        "const bindHoverTooltip = (node, tip, anchorNode) => {"
        "  if (!node || !tip) return;"
        "  if (node.dataset && node.dataset.v2anyqTipBound === '1') return;"
        "  if (node.dataset) node.dataset.v2anyqTipBound = '1';"
        "  const anchor = anchorNode || node;"
        "  node.addEventListener('mouseenter', () => showPopup(anchor, tip));"
        "  node.addEventListener('mousemove', () => showPopup(anchor, tip));"
        "  node.addEventListener('mouseleave', hidePopup);"
        "  node.addEventListener('focus', () => showPopup(anchor, tip));"
        "  node.addEventListener('blur', hidePopup);"
        "};"
        "const attachDropdownTooltip = (root, tip) => {"
        "  if (!root) return;"
        "  const labelNode = pickLabelNode(root);"
        "  if (labelNode) {"
        "    labelNode.querySelectorAll('.param-info-icon').forEach((n) => n.remove());"
        "    labelNode.style.cursor = 'help';"
        "  }"
        "  const nodes = ["
        "    labelNode, root, root.querySelector('.wrap'), root.querySelector('input'),"
        "    root.querySelector('.wrap input'), root.querySelector('button'),"
        "    root.querySelector('[role=\"combobox\"]'), root.querySelector('[data-testid=\"dropdown\"]')"
        "  ];"
        "  const uniq = Array.from(new Set(nodes.filter(Boolean)));"
        "  const anchor = labelNode || root;"
        "  uniq.forEach((n) => bindHoverTooltip(n, tip, anchor));"
        "};"
        "const attachAllIcons = () => {"
        "  for (const [id, tip] of Object.entries(tips)) {"
        "    const el = document.getElementById(id);"
        "    if (!el) continue;"
        "    if (dropdownIds.has(id)) { attachDropdownTooltip(el, tip); continue; }"
        "    const labelNode = pickLabelNode(el);"
        "    addIcon(labelNode, tip);"
        "  }"
        "};"
        "attachAllIcons();"
        "if (!window.__v2anyqInfoIconTimer) {"
        "  window.__v2anyqInfoIconTimer = window.setInterval(attachAllIcons, 800);"
        "}"
        "document.addEventListener('scroll', hidePopup, true);"
        "window.addEventListener('blur', hidePopup);"
        "}"
    )


def _fast_preset_state(*values):
    selected = all(
        actual == expected if isinstance(expected, str)
        else actual is not None and abs(float(actual) - expected) < 1e-8
        for actual, expected in zip(values, _apply_fast_preset())
    )
    return gr.update(value="Apply Fast Preset (Currently selected)" if selected else "Apply Fast Preset", interactive=not selected)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="v2a-nyq UI", css=UI_CSS) as demo:
        with gr.Row(elem_classes="site-header"):
            gr.Markdown("## v2a: Video-to-Audio")
            gr.HTML(
                """<div id="about_us">
<button type="button" popovertarget="about_panel">About us</button>
<div id="about_panel" popover="auto">
<p>This project grew out of a final research project mentored by Roger B. Dannenberg, Professor Emeritus at Carnegie Mellon University, creator of the Nyquist computer music language, and a co-founder of Audacity.</p>
<p>The code was developed primarily by Gordon Yang and Kelly Wong. Jack Chen, Jonathan Ma, and Chelsea Pan contributed to other parts of the project.</p>
</div></div>"""
            )
        gr.Markdown("A music-creation tool based on the Nyquist Engine to make music composition easier and more enjoyable. Select a short video and start generating music!")
        upload_banner = gr.HTML(value="", visible=False)
        state = gr.State({})

        with gr.Row(elem_classes="top-input-row"):
            input_video = gr.File(label="Input Video", file_types=["video"], type="filepath", elem_id="input_video")
            with gr.Column(elem_id="destination_controls"):
                output_dir = gr.Textbox(label="Output Destination", value=str(DEFAULT_OUT_DIR), lines=3, max_lines=3, elem_id="output_dir")
                pick_output_btn = gr.UploadButton(
                    "Choose Output Destination",
                    file_count="directory",
                    type="filepath",
                    elem_id="pick_output",
                )
            nyquist_bin = gr.State("")
            gen_btn = gr.Button("Generate Audio", variant="primary", elem_id="gen_btn")

        task_df = gr.Dataframe(headers=TASK_HEADERS, value=_task_rows(), interactive=False, label="Task Queue", elem_id="task_queue")
        with gr.Accordion("Task controls", open=False):
            with gr.Row():
                cancel_id = gr.Number(label="Task ID to cancel", precision=0, minimum=1, value=None)
                cancel_btn = gr.Button("Cancel task", variant="secondary")
                refresh_btn = gr.Button("Refresh queue", variant="secondary")
            cancel_status = gr.Textbox(label="Cancellation status", interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Parameter Panel")
                fast_preset_btn = gr.Button("Apply Fast Preset (Currently selected)", variant="secondary", interactive=False, elem_id="fast_preset")
                gr.Markdown("Fast preset limits analyzed frames; total time also depends on video decoding and full-length audio/video export.")
                k = gr.Slider(3, 9, value=5, step=1, label="Dominant Color Clusters (k)", elem_id="k")
                method = gr.Dropdown(
                    ["kmeans", "hist"],
                    value="hist",
                    label="Color Feature Method",
                    elem_id="method",
                )
                frame_step = gr.Slider(1, 8, value=2, step=1, label="Frame Sampling Step", elem_id="frame_step")
                frames_per_segment = gr.Slider(2, 24, value=6, step=1, label="Segment Length (Frames)", elem_id="frames_per_segment")
                smooth = gr.Slider(1, 15, value=1, step=1, label="Segment Smoothing Window", elem_id="smooth")
                max_frames = gr.Slider(30, 1200, value=180, step=10, label="Maximum Frames", elem_id="max_frames")
                sensitivity = gr.Slider(0.2, 4.0, value=1.3, step=0.1, label="Visual Sensitivity", elem_id="sensitivity")
                nyquist_timeout = gr.Slider(120, 3600, value=120, step=30, label="Nyquist Timeout (sec)", elem_id="nyquist_timeout")

                tempo_auto = gr.Checkbox(True, label="Auto Tempo", elem_id="tempo_auto")
                tempo = gr.Slider(40, 200, value=120, step=1, label="Tempo (BPM)", interactive=False, elem_id="tempo")

                scale_auto = gr.Checkbox(True, label="Auto Scale/Mode", elem_id="scale_auto")
                scale = gr.Dropdown(SCALES, value="major", label="Scale/Mode", interactive=False, elem_id="scale")
                mood = gr.Dropdown(MOODS, value="none", label="Mood Preset", elem_id="mood")

                gr.Markdown("`Safety correction`  ⟵  `ratio`  ⟶  `Pure visual mapping`")
                mapping_bias = gr.Slider(
                    0.0,
                    1.0,
                    value=0.78,
                    step=0.01,
                    label="Safety Correction / Pure Visual Mapping Ratio",
                    elem_id="mapping_bias",
                )

            with gr.Column():
                gr.Markdown("### Output and Preview")
                video_preview = gr.Video(label="Synthesized Video Preview", interactive=False, elem_id="video_preview")
                audio_preview = gr.Audio(label="Main Mix Preview", type="filepath")
                wav_file = gr.File(label="Exported WAV")
                mp4_file = gr.File(label="Synthesized MP4 (Video + Audio)")
                midi_zip_file = gr.File(label="Tempo-Marked MIDI ZIP (Harmonics / Pad / Bass / Melody)")
        logs = gr.Textbox(label="Run Logs", lines=8, max_lines=16, elem_id="run_logs")

        input_video.upload(
            fn=_on_input_uploaded,
            inputs=[input_video],
            outputs=[upload_banner],
            queue=False,
            show_progress="hidden",
        )
        input_video.clear(
            fn=_hide_upload_banner,
            inputs=[],
            outputs=[upload_banner],
            queue=False,
            show_progress="hidden",
        )
        tempo_auto.change(fn=_toggle_by_auto, inputs=[tempo_auto], outputs=[tempo])
        scale_auto.change(fn=_toggle_by_auto, inputs=[scale_auto], outputs=[scale])
        pick_output_btn.upload(
            fn=_resolve_output_dir_from_picker,
            inputs=[pick_output_btn, output_dir],
            outputs=[output_dir],
        )
        fast_preset_btn.click(
            fn=_apply_fast_preset,
            inputs=[],
            outputs=[k, method, frame_step, frames_per_segment, smooth, max_frames, sensitivity, mapping_bias, nyquist_timeout],
            queue=False,
            show_progress="hidden",
        )

        preset_controls = [k, method, frame_step, frames_per_segment, smooth, max_frames, sensitivity, mapping_bias, nyquist_timeout]
        gr.on(
            triggers=[control.change for control in preset_controls],
            fn=_fast_preset_state,
            inputs=preset_controls,
            outputs=[fast_preset_btn],
            queue=False,
            show_progress="hidden",
        )

        cancel_btn.click(
            fn=_cancel_task,
            inputs=[cancel_id],
            outputs=[cancel_status, task_df],
            queue=False,
        )
        refresh_btn.click(fn=_task_rows, inputs=[], outputs=[task_df], queue=False, show_progress="hidden")

        gen_btn.click(
            fn=_run_generator,
            inputs=[
                input_video,
                output_dir,
                nyquist_bin,
                k,
                method,
                frame_step,
                frames_per_segment,
                smooth,
                tempo_auto,
                tempo,
                scale_auto,
                scale,
                mood,
                sensitivity,
                max_frames,
                mapping_bias,
                nyquist_timeout,
                state,
            ],
            outputs=[video_preview, audio_preview, wav_file, mp4_file, midi_zip_file, logs, task_df, state],
        )

        demo.load(fn=None, js=_tooltip_js())

    return demo.queue(max_size=32, default_concurrency_limit=2)


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", share=False)

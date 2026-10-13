# v2a: Video-to-Audio

v2a turns the colors, motion, and visual changes in a short video into an algorithmic musical composition. It extracts visual features in Python, maps them into musical structure, renders the result with the Nyquist engine, and exports synchronized audio/video and MIDI stems through a Gradio interface.

## Features

- Browser-based Gradio interface
- Fast preset for shorter processing times
- Automatic tempo and scale selection
- Adjustable visual sensitivity, segmentation, smoothing, and mapping bias
- Nyquist composition and audio synthesis
- WAV, synchronized MP4, and four tempo-marked MIDI stem exports
- Task queue with estimated completion time and cancellation controls

## Requirements

- Python 3.10 or newer
- [Nyquist](https://www.cs.cmu.edu/~music/nyquist/) with the `ny` or `nyquist` executable available on `PATH`
- FFmpeg, supplied automatically by `imageio-ffmpeg` on supported platforms or available on `PATH`

Nyquist runs on the machine hosting this application. People visiting a hosted instance only need a web browser.

## Installation

```bash
git clone https://github.com/VectorScore/v2a.git
cd v2a
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Run the website

```bash
python ui_gradio.py
```

Open the local URL shown in the terminal, usually `http://127.0.0.1:7860`.

## Command-line use

```bash
python convert-tool.py input/video.mp4 output/result.wav --synth-output
```

Use `python convert-tool.py --help` for all options.

## Repository structure

```text
v2a/
├── convert-tool.py          # Video analysis and conversion pipeline
├── composition_engine.py    # Music planning and MIDI event generation
├── media_sync.py            # FFmpeg-based duration matching and muxing
├── simple_kmeans.py         # Lightweight NumPy k-means implementation
├── ui_gradio.py             # Web interface and task queue
├── v2a_nyq_engine.lsp       # Nyquist synthesis engine
├── v2a_nyq_engine.sal       # Nyquist/SAL engine source
├── docs/                    # Technical explanation
├── input/                   # Local input files; contents are ignored
└── output/                  # Generated files; contents are ignored
```

## Notes

- Large videos take longer to decode even when the fast preset limits the number of analyzed frames.
- The application fails with a clear error when Nyquist cannot be found. There is currently no alternative synthesis backend.
- Input videos and generated outputs are intentionally excluded from Git.

## Credits

This project grew out of a final research project mentored by Roger B. Dannenberg, Professor Emeritus at Carnegie Mellon University, creator of the Nyquist computer music language, and a co-founder of Audacity.

The code was developed primarily by Gordon Yang and Kelly Wong. Jack Chen, Jonathan Ma, and Chelsea Pan contributed to other parts of the project.

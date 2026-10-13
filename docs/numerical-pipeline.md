v2a-nyq End-to-End Numerical Walkthrough

Purpose
This document explains the full v2a-nyq pipeline numerically, from video frames to final WAV output.
The goal is to make every stage concrete: what values exist, how they are computed, and how parameters change the result.

==================================================
1) Input Video -> Sampled Frames
==================================================

A video is decoded frame by frame.
Important parameters at this stage:
- frame_step: keep every Nth frame
- max_frames: hard cap on sampled frames
- max_size: resize longest frame side

Example:
- Original FPS = 29.97
- frame_step = 2
- max_frames = 300

Then:
- Effective sampled frame interval = frame_step / fps = 2 / 29.97 = 0.0667 seconds
- Effective sampled FPS is approximately 14.985

If the source video is 20 seconds long:
- Total original frames ~ 599
- With frame_step=2, sampled frames ~ 300
- max_frames=300 means we keep exactly those 300 and stop.

==================================================
2) RGB Frame -> HSV Numeric Matrix
==================================================

Each sampled frame is converted from RGB to HSV and normalized to [0, 1].
For each pixel:
- H in [0,1], where 0 and 1 represent the same hue (circular axis)
- S in [0,1]
- V in [0,1]

Simple pixel examples (conceptual):
- RGB(255, 0, 0) -> HSV ~ (0.00, 1.00, 1.00)
- RGB(0, 255, 0) -> HSV ~ (0.33, 1.00, 1.00)
- RGB(0, 0, 255) -> HSV ~ (0.67, 1.00, 1.00)
- RGB(128, 128, 128) -> HSV ~ (any hue, 0.00, 0.50)

After conversion, each frame becomes a 3D numeric array of shape:
- height x width x 3
with channels [H, S, V].

==================================================
3) Segment Construction
==================================================

Frames are grouped into segments using:
- frames_per_segment

Example:
- Sampled frames = 300
- frames_per_segment = 5

Number of segments:
- 300 / 5 = 60 segments

Duration of one segment before Nyquist quantization:
- segment_duration_raw = frames_in_segment * (frame_step / fps)
- = 5 * (2/29.97)
- = 0.3335 seconds

So each segment carries about one-third of a second of visual information.

==================================================
4) Pixel Sampling per Segment
==================================================

Inside each segment, all pixels from all segment frames are merged.
Then up to "sample" pixels are randomly selected (without replacement).

Example:
- Resized frame = 512 x 288 = 147,456 pixels
- 5 frames per segment -> 737,280 HSV pixels total
- sample = 60,000
- Randomly choose 60,000 rows from 737,280

This keeps compute bounded and stable.

==================================================
5) Segment Statistics (method = kmeans or hist)
==================================================

Each segment produces 8 numbers:
1. hue_mean
2. sat_mean
3. val_mean
4. hue_var
5. sat_var
6. val_var
7. motion
8. duration_sec

--------------------------------------------------
5A) KMeans variant example (method = kmeans)
--------------------------------------------------

Suppose KMeans finds 3 cluster centers in one segment:
- C1 = (H=0.08, S=0.72, V=0.84), count=5000
- C2 = (H=0.34, S=0.55, V=0.40), count=3000
- C3 = (H=0.90, S=0.20, V=0.92), count=2000

Weights:
- w1=0.50, w2=0.30, w3=0.20

sat_mean:
- 0.50*0.72 + 0.30*0.55 + 0.20*0.20
- = 0.360 + 0.165 + 0.040
- = 0.565

val_mean:
- 0.50*0.84 + 0.30*0.40 + 0.20*0.92
- = 0.420 + 0.120 + 0.184
- = 0.724

sat_var (weighted):
- 0.50*(0.72-0.565)^2 + 0.30*(0.55-0.565)^2 + 0.20*(0.20-0.565)^2
- = 0.0120125 + 0.0000675 + 0.026645
- = 0.038725

val_var (weighted):
- 0.50*(0.84-0.724)^2 + 0.30*(0.40-0.724)^2 + 0.20*(0.92-0.724)^2
- = 0.006728 + 0.0314928 + 0.0076832
- = 0.045904

hue_mean and hue_var use circular math because hue wraps.
Using vector average on the unit circle for H=[0.08,0.34,0.90] with weights [0.50,0.30,0.20], a valid outcome is approximately:
- hue_mean ~ 0.113
- hue_var ~ 0.422

--------------------------------------------------
5B) Histogram-style fallback (method = hist)
--------------------------------------------------

If not using KMeans, means/variances come directly from sampled pixels.
- sat_mean = arithmetic mean of S values
- val_mean = arithmetic mean of V values
- sat_var = variance of S values
- val_var = variance of V values
- hue_mean/hue_var still use circular formulas

==================================================
6) Motion per Segment
==================================================

Motion is computed from per-frame mean HSV inside each segment.

Step 1: compute each frame mean:
- h_mean (circular mean of pixel hues)
- s_mean (mean S)
- v_mean (mean V)

Step 2: compute pairwise differences between consecutive frame means:
- h_diff = circular_distance(h_t, h_{t-1}) = min(|a-b|, 1-|a-b|)
- s_diff = |s_t - s_{t-1}|
- v_diff = |v_t - v_{t-1}|
- pair_diff = (h_diff + s_diff + v_diff)/3

Step 3: motion = mean(pair_diff values), clipped to [0,1]

Example with 3 frames:
- F0 mean = (h=0.10, s=0.55, v=0.65)
- F1 mean = (h=0.16, s=0.58, v=0.63)
- F2 mean = (h=0.14, s=0.62, v=0.68)

F0->F1:
- h_diff=0.06, s_diff=0.03, v_diff=0.02
- pair_diff=(0.06+0.03+0.02)/3=0.0367

F1->F2:
- h_diff=0.02, s_diff=0.04, v_diff=0.05
- pair_diff=(0.02+0.04+0.05)/3=0.0367

motion:
- (0.0367 + 0.0367)/2 = 0.0367

Interpretation:
- 0.0367 is low motion (visually stable segment)

==================================================
7) Segment Smoothing Window
==================================================

After raw segment features are computed, smoothing is applied across neighboring segments.

Parameter:
- smooth = window size
- radius = smooth // 2

For each segment i:
- take block from [i-radius, i+radius]
- average sat_mean, val_mean, hue_var, sat_var, val_var, motion
- hue_mean uses circular mean over neighboring hue_means
- duration_sec is kept from the original segment i

Example (motion only) with window=3:
Raw segment motion:
- [0.10, 0.40, 0.90, 0.20, 0.30]

Smoothed:
- seg0 uses [0.10,0.40] -> 0.25
- seg1 uses [0.10,0.40,0.90] -> 0.4667
- seg2 uses [0.40,0.90,0.20] -> 0.50
- seg3 uses [0.90,0.20,0.30] -> 0.4667
- seg4 uses [0.20,0.30] -> 0.25

Result:
- [0.25, 0.4667, 0.50, 0.4667, 0.25]

==================================================
8) Python -> Nyquist Data Transfer
==================================================

Each segment is serialized as one row:
(hue_mean sat_mean val_mean hue_var sat_var val_var motion duration_sec)

Example row:
(0.11300000 0.56500000 0.72400000 0.42200000 0.03872500 0.04590400 0.41000000 0.33350000)

Nyquist receives a list of these rows as *v2a-segments*.

==================================================
9) Auto Scale Selection in Nyquist
==================================================

Nyquist computes global averages over segments:
- avg_sat = mean(column sat_mean)
- avg_val = mean(column val_mean)
- avg_hue_var = mean(column hue_var)
- avg_motion = mean(column motion)

Decision order:
1) if mood=cheerful -> major
2) else if mood=epic -> mixolydian
3) else if avg_motion > 0.35 -> dorian
4) else if avg_sat > 0.58 -> major
5) else if avg_val < 0.36 -> minor
6) else if avg_hue_var > 0.24 -> mixolydian
7) else -> pentatonic

Example:
- mood=none
- avg_motion=0.41
- avg_sat=0.57
- avg_val=0.62
- avg_hue_var=0.18

Rule 3 triggers first:
- chosen scale = dorian

==================================================
10) Auto BPM in Nyquist
==================================================

Nyquist computes:
- motion = mean(segment motion)
- sat = mean(segment sat_mean)
- val_var = mean(segment val_var)

Base formula:
- base = 72 + 88*motion + 26*sat + 18*val_var

Mood scaling:
- cheerful: base * 1.10
- epic: base * 0.92
- none: base

Final BPM:
- bpm = clamp(round(base_scaled), 42, 196)

Example:
- motion=0.41
- sat=0.57
- val_var=0.05

base:
- 72 + 88*0.41 + 26*0.57 + 18*0.05
- = 72 + 36.08 + 14.82 + 0.90
- = 123.80

Outputs:
- mood none -> 124 BPM
- mood cheerful -> round(123.80*1.10)=136 BPM
- mood epic -> round(123.80*0.92)=114 BPM

==================================================
11) Per-Segment Time Quantization in Nyquist
==================================================

Given a segment row and BPM:
- raw_sec = max(0.05, duration_sec)
- beat_sec = 60 / bpm
- beats = raw_sec / beat_sec

Grid depends on motion:
- if motion > 0.45 -> grid = 0.25 beat
- else -> grid = 0.50 beat

Then:
- steps = max(1, round(beats / grid))
- quantized_beats = steps * grid
- final_segment_duration = quantized_beats * beat_sec

Example A (lower motion):
- bpm=124 => beat_sec=0.4839
- raw_sec=0.3335
- beats=0.3335/0.4839=0.689
- motion=0.41 -> grid=0.5
- beats/grid=1.378 -> round=1
- quantized_beats=0.5
- final_duration=0.5*0.4839=0.2419 sec

Example B (higher motion):
- same raw_sec=0.3335, bpm=124
- motion=0.72 -> grid=0.25
- beats/grid=2.756 -> round=3
- quantized_beats=0.75
- final_duration=0.75*0.4839=0.3629 sec

Important effect:
- High-motion segments can become rhythmically finer and sometimes slightly longer after quantization.

==================================================
12) Segment -> Scale Degree -> MIDI Pitches
==================================================

Let chosen scale be dorian:
- intervals = [0,2,3,5,7,9,10]
- length = 7

For each segment:
- degree = floor(hue_mean * 7) mod 7

Example:
- hue_mean=0.61
- degree=floor(0.61*7)=4

Root/third/fifth for harmony:
- root  = scale_degree(4) from base MIDI 48 -> 55
- third = scale_degree(6) from base MIDI 48 -> 58
- fifth = scale_degree(8) from base MIDI 48 -> 62

Bass:
- bass = root - 12 = 43

So this segment harmonically centers around G-like color in this register:
- root 55, third 58, fifth 62, bass 43

==================================================
13) Mapping-Bias and Sensitivity -> Melodic Jump
==================================================

Inputs:
- bias = clamp(mapping_bias, 0,1)
- safe_weight = 1 - bias
- motion
- sensitivity

Motion gain:
- motion_gain = 1 + sensitivity * (0.20 + 0.80*bias)

Melody jump amount:
- jump = floor(motion * motion_gain * 4)

Example:
- mapping_bias=0.65
- sensitivity=1.30
- motion=0.41

motion_gain:
- 1 + 1.30*(0.20 + 0.80*0.65)
- 1 + 1.30*(0.20 + 0.52)
- 1 + 1.30*0.72
- 1 + 0.936
- 1.936

jump:
- floor(0.41 * 1.936 * 4)
- floor(3.175)
- 3

Melody scale degree index:
- melody_degree = degree + 7 + jump
- with degree=4 -> 4 + 7 + 3 = 14

Melody base MIDI register:
- base = 60 + register_shift
- register_shift = +12 for cheerful, -3 for epic, 0 otherwise

If mood=none:
- base=60
- degree 14 in 7-note scale means oct=2, idx=0
- melody MIDI = 60 + 0 + 24 = 84

==================================================
14) Amplitude and Duration per Voice
==================================================

Given segment S,V,motion,bias:

Master amplitude:
- amp = 0.22 * (0.35 + 0.65*V)

Voice gains:
- h_amp = amp * (0.55 + 0.45*safe_weight)
- b_amp = amp * (0.60 + 0.40*S)
- m_amp = amp * (0.50 + 0.50*motion)

Melody duration inside segment:
- mel_dur = max(0.08, dur*(0.34 + 0.46*(1-motion)))

Sparkle duration:
- sparkle_dur = max(0.06, dur*0.18)

Example values:
- V=0.68, S=0.62, motion=0.41, bias=0.65, safe_weight=0.35
- dur=0.2419 sec

amp:
- 0.22*(0.35+0.65*0.68)
- 0.22*(0.35+0.442)
- 0.22*0.792
- 0.17424

h_amp:
- 0.17424*(0.55+0.45*0.35)
- 0.17424*0.7075
- 0.12328

b_amp:
- 0.17424*(0.60+0.40*0.62)
- 0.17424*0.848
- 0.14776

m_amp:
- 0.17424*(0.50+0.50*0.41)
- 0.17424*0.705
- 0.12284

mel_dur:
- max(0.08, 0.2419*(0.34 + 0.46*(1-0.41)))
- max(0.08, 0.2419*(0.34 + 0.2714))
- max(0.08, 0.2419*0.6114)
- max(0.08, 0.1479)
- 0.1479 sec

==================================================
15) Segment Sound Layout in Time
==================================================

For each segment duration dur:
- Harmony triad starts at segment start and lasts dur
- Bass note starts at segment start and lasts dur
- Melody note starts at 0.10*dur offset
- Sparkle note is added only if motion > 0.52, starts at 0.58*dur, and is one octave above melody

Sparkle condition example:
- if motion = 0.41 -> no sparkle
- if motion = 0.73 -> sparkle added

If dur=0.3629 and motion=0.73:
- melody starts at 0.0363 sec from segment start
- sparkle starts at 0.2105 sec from segment start

==================================================
16) Whole Piece Assembly
==================================================

Segments are composed sequentially:
- Segment 0 starts at t=0
- Segment 1 starts at t = dur0
- Segment 2 starts at t = dur0 + dur1
- etc.

Inside each segment, the 4-layer sound (harmony+bass+melody+optional sparkle) is mixed.
Across segments, these mixed blocks are concatenated on the global timeline.

Final stage:
- global sample rate is set (default 44100)
- final piece is scaled by 0.92 headroom multiplier
- saved to output WAV path

==================================================
17) Full Mini End-to-End Example (Compact)
==================================================

Assume one segment after smoothing:
- hue_mean=0.61
- sat_mean=0.62
- val_mean=0.68
- hue_var=0.18
- sat_var=0.03
- val_var=0.05
- motion=0.41
- duration_sec=0.3335

Assume global auto decisions:
- scale = dorian
- bpm = 124
- mood = none
- mapping_bias = 0.65
- sensitivity = 1.30

Computed:
- beat_sec = 60/124 = 0.4839
- segment dur quantized = 0.2419 sec
- degree = floor(0.61*7)=4
- harmony notes = [55,58,62]
- bass note = 43
- motion_gain = 1.936
- jump = 3
- melody note = 84
- amp split -> h_amp=0.1233, b_amp=0.1478, m_amp=0.1228
- melody length = 0.1479 sec
- sparkle absent (motion <= 0.52)

Audible result for this segment:
- Short quantized segment with stable triad + bass foundation
- One melody tone entering slightly after segment start
- Moderate loudness and no extra sparkle ornament

==================================================
18) What Each Main Parameter Numerically Controls
==================================================

frame_step:
- Multiplies per-sampled-frame time by frame_step/fps
- Larger value -> fewer frames sampled -> less temporal detail

frames_per_segment:
- Multiplies raw segment duration and pixel pool size
- Larger value -> coarser musical update rate

method and k:
- kmeans uses weighted cluster centers; k changes palette granularity
- hist-style mode uses direct pixel distributions

smooth:
- Local averaging across neighboring segments
- Larger value lowers jitter in motion/means/variances

mapping_bias:
- Raises motion_gain influence and reduces harmony safety weight
- Higher values can increase melodic jump volatility

sensitivity:
- Linear multiplier in motion_gain formula
- Higher values increase jump response to the same motion

tempo override / scale override:
- Replaces auto decisions directly

==================================================
19) Final Practical Interpretation
==================================================

The pipeline is effectively a numeric transformation chain:

Video frames
-> HSV tensors
-> segment statistics + motion
-> optional cross-segment smoothing
-> per-segment numeric rows
-> Nyquist global decisions (scale, BPM)
-> per-segment pitch/amplitude/duration mapping
-> layered synthesis events on timeline
-> WAV render.

The strongest drivers of perceived musical behavior are usually:
1) motion (rhythmic granularity and melodic jump)
2) hue_mean (scale degree location)
3) val_mean and sat_mean (amplitude balance)
4) smooth window (stability vs reactivity)
5) mapping_bias + sensitivity (expressiveness intensity)


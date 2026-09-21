# SOKE-ASL: Real-Time Speech-to-ASL Sign Language Generation

**Base model:** [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](https://arxiv.org/pdf/2411.17799) (Zuo et al., **ICCV 2025**)
---
## 1. Objective

SOKE is a text-to-sign generator trained and evaluated on offline, batch data across three sign languages (How2Sign/ASL, CSL-Daily, Phoenix-2014T), using pre-fit SMPL-X poses. This project has two goals:

1. **Fine-tune SOKE specifically on ASL** (How2Sign) rather than relying on the multilingual checkpoint, and measure motion-reconstruction accuracy after fine-tuning.
2. **Extend SOKE into a live speech-to-sign pipeline** by prepending speech recognition, wrapping the generator in an endpoint-detection state machine, and deploying inference and rendering under real-time latency constraints on a Raspberry Pi 4B.

SOKE's own paper only addresses (1) — text in, motion out, offline. Component (2) is not part of the original work; it is the system-engineering contribution layered on top for this project.

---

## 2. System Architecture

```
Microphone → VAD segmentation → Faster-Whisper (ASR) → text normalization
    → mBART/SOKE (motion-token generation) → 3× VQ-VAE decoders
    → SMPL-X (mesh + joints) → pose compression → Raspberry Pi client → skeleton render
```

### 2.1. Speech Recognition (Faster-Whisper)

Audio is transcribed with Faster-Whisper, a CTranslate2-optimized reimplementation of Whisper, chosen to decouple ASR from motion generation into two independently verifiable stages rather than mapping audio directly to motion. Running under real-time constraints, the model uses a reduced size, quantized weights, and a small beam width to trade a small amount of accuracy for lower latency. Recognized text is normalized (filler-word removal, whitespace cleanup, repetition collapsing) before being passed to the generator.

### 2.2. Decoupled Tokenizer (VQ-VAE)

A language model operates on discrete tokens; body motion is continuous 3D coordinates. SOKE bridges this gap with a **decoupled tokenizer** — three independent VQ-VAEs, one per body region:

| Tokenizer | Input | Output | Role |
|---|---|---|---|
| Body VQ-VAE | Body token | Upper-body, head, jaw, expression params | Reconstructs torso motion |
| Left-hand VQ-VAE | Left-hand token | 45 parameters | Reconstructs left-hand joints |
| Right-hand VQ-VAE | Right-hand token | 45 parameters | Reconstructs right-hand joints |

Hands are tokenized separately from the torso because most of ASL's lexical information lives in hand shape and finger configuration — a single shared tokenizer would blur that detail against coarser body motion. Each token corresponds to roughly 4 frames; an L-token sequence decodes to a motion sequence of length T ≈ 4L, each frame carrying 133 parameters.

```bash
python -m train --cfg configs/deto.yaml --nodebug   # tokenizer training
python -m test  --cfg configs/deto.yaml --nodebug   # tokenizer inference
```

### 2.3. Autoregressive Generator (mBART-large-cc25)

A pretrained multilingual mBART backbone consumes normalized text and autoregressively generates motion-token sequences — the model predicts *discrete motion codes* rather than raw joint coordinates, letting it inherit a language model's sequence-modeling capacity for word order, semantics, and temporal progression.

Two departures from prior flatten-and-decode-one-token-at-a-time approaches:

- **Multi-head decoding** — body / left-hand / right-hand tokens are predicted simultaneously through separate decoding heads, cutting the number of decode steps by roughly two-thirds while still fusing cross-part information.
- **Retrieval-enhanced generation** — keywords in the input sentence are used to retrieve word-level sign motion tokens from an external sign dictionary, which are fed to the decoder as auxiliary conditioning. This improves accuracy specifically on rare words, numerals, and hand shapes the base model would otherwise under-generate.

```bash
python -m get_motion_code --cfg configs/soke.yaml --nodebug
python -m train            --cfg configs/soke.yaml --nodebug
python -m test              --cfg configs/soke.yaml --task t2m
```

### 2.4. Body Rendering (SMPL-X)

The `[T, 133]` parameter sequence is passed through SMPL-X to produce `vertices [T, 10475, 3]` (mesh for avatar rendering) and `joints [T, N, 3]` (skeleton — used both for visualization and for evaluation).

### 2.5. Endpoint Detection (Speech-side State Machine)

Because SOKE's paper assumes complete text as input, utterance boundaries must be detected before any request is sent to the generator. This is handled by an energy-based VAD running client-side:

| Stage | Rule |
|---|---|
| Frame size | 100 ms |
| Noise calibration | 0.8 s ambient sampling at session start |
| Utterance finalize | 0.45 s of silence, or 7 s hard cap |
| Rejection filters | discard utterances < 0.25 s total, or < 0.20 s of actual speech |

The 7 s cap exists purely to bound worst-case latency — without it, a long uninterrupted sentence would delay generation indefinitely. Rejection filters run before any network call, so false triggers never reach the GPU-side pipeline.

---

## 3. Data

| Attribute | Value |
|---|---|
| Dataset | How2Sign (ASL subset — distinct from SOKE's original multilingual training set) |
| Scale | ~80 hours of video, ~35,000 sentences, with English transcripts, gloss, audio, and body/hand/face pose |
| Video source | Green-screen RGB, frontal view |
| SMPL-X poses | Pre-fit, from the project homepage (no pose-fitting run in this repo) |
| Split | Per How2Sign's original split files |

---

## 4. Evaluation Metric

Evaluation uses **DTW-MPJPE**:

- **MPJPE** (Mean Per Joint Position Error): mean Euclidean error between predicted and reference 3D joint coordinates, after root-relative alignment.
- **DTW** (Dynamic Time Warping): two motion sequences can convey the same content at different signing speeds. DTW finds an optimal alignment path between predicted and reference sequences before averaging joint error along that path, so tempo mismatches aren't penalized as content errors.

$$
\mathrm{DTW\text{-}MPJPE}
=
\frac{1}{|P|}
\sum_{(i,j) \in P} d(x_i, y_j)
$$

where:
- $P$ is the set of index pairs on the optimal DTW alignment path.
- $d(x_i, y_j)$ is the per-joint Euclidean error between predicted frame $i$ and reference frame $j$.

---

## 5. Results

| Body part | DTW-MPJPE (test) |
|---|---|
| Upper body | 14.8 mm |
| Left hand | 22.5 mm |
| Right hand | 23.1 mm |
| **Average** | **18.7 mm (val) / 20.3 mm (test)** |

**Observation:** hand error runs ~55% higher than body error, consistent with the decoupled-tokenizer design rationale — hands have more degrees of freedom and finer articulation than torso motion, so even with a dedicated VQ-VAE per hand, they remain the harder region to reconstruct precisely.

**Metric limitation:** DTW-MPJPE measures geometric deviation after temporal alignment only — it does **not** verify whether the generated sign is semantically correct. A motion that is a few millimeters off but shape-plausible could still convey the wrong meaning if it lands near a minimal pair in ASL.

### End-to-end latency (system-level)

| Metric | Baseline | Optimized | Improvement |
|---|---|---|---|
| Mean latency (VAD finalize → first rendered frame) | 2.146 s | 1.562 s | −27.2% |
| P95 latency | 3.284 s | 2.421 s | −26.3% |

Optimizations contributing to this reduction: bounded audio queue (~2 s), a hard cap of 2 in-flight generation requests, raw-WAV transport (no multipart overhead), a reused HTTP session with gzip-compressed pose payloads, a (text, language)-keyed pose cache to skip re-inference on repeated utterances, and FPS-capped skeleton rendering (16–24 FPS, dirty-frame skipping) on the Pi 4B client.

---
## 6. Citation

```bibtex
@inproceedings{zuo2025soke,
    title={Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator},
    author={Zuo, Ronglai and Potamias, Rolandos Alexandros and Ververas, Evangelos and Deng, Jiankang and Zafeiriou, Stefanos},
    booktitle={ICCV},
    year={2025}
}
```

Built on: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT/), [ProgressiveTransformer](https://github.com/BenSaunders27/ProgressiveTransformersSLP), [WiLoR](https://github.com/rolpotamias/WiLoR), [OSX](https://github.com/IDEA-Research/OSX/).

# Generated from: f2-transcription-translation.ipynb
# Converted at: 2026-08-22T17:44:01.446Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Audio and video transcription
# 
# This notebook starts the recorded-media system from scratch. It accepts a **completed microphone recording, webcam recording, or uploaded audio/video file**. There is no live streaming and no application-level chunk-and-stitch pipeline.
# 
# The completed media file is passed to the **official OpenAI PyTorch Whisper implementation** as one input. Only the reference `openai-whisper` implementation is installed and used. Translation is **text-to-text only**: it produces normal written English or Hindi and never synthesizes an artificial voice.
# 
# ```text
# Completed audio/video
#         │
#         ├─ inspect channels ── two real call channels? ── yes ─> transcribe each channel
#         │                                      │
#         │                                      no
#         │                                      ↓
#         ├─ preserve one clean 16 kHz ASR + speaker-preserving diarization copy
#         ├─ create enhanced ASR audio only if quality evidence requests a retry
#         │                                      ↓
#         ├─ Whisper large-v3 auto + English + Hindi coverage passes
#         │                                      ↓
#         ├─ confidence/quality selection ─> NeMo speaker and overlap timeline
#         │                                      ↓
#         ├─ Stage 1 returns the finalized timestamped speaker transcript
#         └─ Stage 2 (user starts it) ─> written English ↔ Hindi translation
# ```
# 
# ## Accuracy rules used here
# 
# - **Stereo call recordings are not downmixed blindly.** Separate call legs require low correlation plus alternating per-channel energy dominance. Each untouched isolated leg remains an ASR candidate; crosstalk-reduced audio is only an additional comparison. Ordinary stereo ambience is downmixed and diarized.
# - **Mono audio reuses the minimally resampled speaker-preserving path for diarization.** ASR denoising/compression is never allowed to change the acoustic speaker identity seen by NeMo.
# - **Preprocessing never destroys the original evidence.** Whisper starts with a minimally resampled copy and creates an ASR-only enhanced copy only when a retry is justified. Silence is never removed, so every timestamp stays on the original recording clock.
# - **NeMo output remains independent of Whisper.** NVIDIA's recommended call-style onset/offset rules are enabled. Speaker assignment and exports preserve the complete NeMo timeline; a separate Whisper-gated copy is used only for compact presentation.
# - **Uncertain attribution stays uncertain.** A word is not forced to the nearest speaker when NeMo has insufficient time overlap. Overlapping and low-confidence assignments are explicitly marked for review.
# - **Later evidence can repair an earlier label.** After the full file is processed, an uncertain non-overlap fragment is reassigned only when NeMo's acoustic candidate agrees with nearby confirmed speech, including a later turn. Conflicting evidence remains uncertain; speaker IDs are never treated as verified human identities.
# - **Balanced mode covers the required languages.** It compares auto, Hindi and English whole-file decodes, adds the preserved crosstalk-reduced leg for separated stereo calls, and adds enhanced audio only when the signal is weak. Maximum-accuracy mode compares all language/audio variants deliberately.
# - **Whisper is restricted to the required language choices.** Auto, Hindi and English passes reduce the risk of Hindi being rendered as Urdu/Arabic script.
# - **No domain prompt is injected.** The transcript is driven by the conversation, not by loan/EMI/KYC vocabulary.
# - **Translation does not use arbitrary word chunks.** Each finalized timestamped speaker turn is translated once and keeps the same timestamp.
# - **Translation is a separate second stage.** IndicTrans2 handles normal complete turns. A targeted Whisper speech-translation fallback can re-decode only a failed finalized Hindi/Hinglish turn from its matching original-audio interval.
# - **Post-processing is conservative.** It clamps invalid timestamps, removes only overlapping near-duplicates, preserves repetitions and disfluencies as spoken, and merges adjacent turns only when speaker, overlap and uncertainty state agree.
# - **Temporary GPU allocations are cleared after every request.** Cleanup runs in `finally`, including failed jobs, while model weights remain loaded for the next file.
# 
# Run the cells from top to bottom on a Colab GPU runtime. For production evaluation, use consented/de-identified recordings and compare against human reference transcripts.


# ## Step 1 — Why these models and settings
# 
# ### Speech recognition
# 
# - **Whisper large-v3** is the requested multilingual checkpoint. Its model card reports a 10–20% error reduction over large-v2 across many languages.
# - A public Hindi benchmark shown on the model card reports **26.8% WER**. That is a benchmark result, not a guarantee for your calls: telephony compression, accents, crosstalk and labels all change WER.
# - Whisper's acoustic receptive field is 30 seconds. For long files it must move through internal windows. We use sequential long-form decoding and do not create independent user-visible chunks. The model card recommends sequential decoding when accuracy matters and reports up to about **0.5 WER point** advantage over independent chunked decoding.
# 
# ### Speaker diarization
# 
# - **NVIDIA NeMo Streaming Sortformer 4-speaker v2.1** is used on completed mono recordings with its longest/highest-context preset. NVIDIA reports **5.65% DER on 2-speaker CALLHOME** at the 30.4-second latency configuration and includes overlapping speech in evaluation.
# - NeMo's documented post-processing values are applied (`onset=0.64`, `offset=0.74`, minimum speech `0.10 s`, minimum gap `0.15 s`). NVIDIA documents that default post-processing is otherwise bypassed and recommends `batch_size=1` for the longest, highest-accuracy inference window.
# - Speaker identity reuses the minimally resampled waveform. The aggressive ASR-enhanced waveform is deliberately excluded because denoising and dynamic normalization can alter speaker cues.
# - DER and WER are different: DER measures speaker-time errors; WER measures transcription word errors. Results from different datasets/collars are not directly comparable.
# 
# ### What must be measured on your data
# 
# The notebook includes WER/CER evaluation. Build a representative, human-transcribed test set containing Hindi, English, Hinglish, low-volume speech, background noise and overlaps. A model is acceptable only if it meets the target on that held-out set.
# 
# Primary references:
# 
# - https://huggingface.co/openai/whisper-large-v3
# - https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
# - https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html


# ## Step 2 — Install dependencies


# Install once in a fresh Colab GPU runtime.
!apt-get -qq update
!apt-get -qq install -y ffmpeg libsndfile1
!pip -q install --upgrade \
  "numpy==1.26.4" \
  "openai-whisper" \
  "gradio==5.49.1" \
  "soundfile==0.13.1" \
  "transformers==4.57.3" \
  "sentencepiece==0.2.1" \
  "sacremoses==0.1.1" \
  "IndicTransToolkit==1.1.1" \
  "nemo_toolkit[asr]==2.7.3" \
  "jiwer==3.1.0" \
  "flashtext==2.7"

from importlib.metadata import version
from flashtext import KeywordProcessor  # Import statement for code usage

for package_name in (
    "openai-whisper", "nemo-toolkit", "jiwer", "gradio", "transformers", "flashtext"
):
    print(f"{package_name}: {version(package_name)}")

print("Installation verified.")

# ## Step 3 — Configure the offline recorded-media pipeline
# 
# The T4 GPU is used for accuracy-first inference.


import gc  #This module provides access to the garbage collector for reference cycles.
import json
import math
import re  #regular expression matching
import shutil #coping and archieve files
import subprocess # Subprocesses with accessible I/O streams
# This module allows you to spawn processes, connect to their
# input/output/error pipes, and obtain their return codes.
import threading
import time
import unicodedata
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import gradio as gr
import jiwer
import numpy as np
import soundfile as sf  # read audio files
import torch
import transformers
import whisper

if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU in Runtime → Change runtime type, then rerun this cell.")

# SAMPLE_RATE = 16_000 # target audio sampling rate 16khz
# MIN_MEDIA_SECONDS = 0.5 # min audio len 0.5
# MAX_MEDIA_SECONDS = 60 * 60
# ASR_BEAM_SIZE = 1 
# ASR_RETRY_BEAM_SIZE = 2
# TRANSLATION_BEAM_SIZE = 5 # translation decoder maintains top 5 candidate sentences simultaneously before selecting final sequence
# TEXT_TRANSLATION_BATCH_SIZE = 8 # group up to 8 finalized speaker turn sentences into tensor passed to GPU during stage 2 translation
# MAX_DIARIZATION_SPEAKERS = 4 # maximum no of distinct speakers voiceprints
# STEREO_SEPARATION_CORRELATION = 0.80
# STEREO_DOMINANCE_DB = 9.0
# WHISPER_TRANSLATION_BEAM_SIZE = 5


#improve results
SAMPLE_RATE = 16_000
MIN_MEDIA_SECONDS = 0.5
MAX_MEDIA_SECONDS = 60 * 60
ASR_BEAM_SIZE = 1
ASR_RETRY_BEAM_SIZE = 5         # Updated: Maximizes retry recovery on accented speech
TRANSLATION_BEAM_SIZE = 5
TEXT_TRANSLATION_BATCH_SIZE = 16 # Updated: Speeds up Stage 2 batch translation on GPU
MAX_DIARIZATION_SPEAKERS = 4
STEREO_SEPARATION_CORRELATION = 0.80
STEREO_DOMINANCE_DB = 9.0
WHISPER_TRANSLATION_BEAM_SIZE = 5


ASR_MODEL_ID = "large-v3"
DIARIZATION_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
EN_HI_MODEL_ID = "naklitechie/indictrans2-en-indic-dist-200M"
HI_EN_MODEL_ID = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
EN_HI_REVISION = "a814dab1ae6e4ee4c7d785b7e1dcb0ac8e36bcd6"
HI_EN_REVISION = "00213ee82929050694b162123bb26a8dc177cdae"

MODEL_ROOT = Path("/content/accuracy_models")
OUTPUT_ROOT = Path("/content/accuracy_outputs")
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Only one completed recording uses the GPU at a time.
INFERENCE_LOCK = threading.Lock()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print("GPU:", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
print("Output folder:", OUTPUT_ROOT)

# ## Step 4 — Load Whisper, NeMo and translation models once
# 
# The exact checkpoints are cached under `/content/accuracy_models` for this runtime.


from transformers import AutoModelForSeq2SeqLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import transformers.tokenization_utils as tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
    tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

from IndicTransToolkit.processor import IndicProcessor
from nemo.collections.asr.models import SortformerEncLabelModel

for loaded_name in (
    "ASR_MODEL", "DIARIZATION_MODEL", "EN_HI_MODEL", "HI_EN_MODEL",
    "EN_HI_TOKENIZER", "HI_EN_TOKENIZER",
):
    if loaded_name in globals():
        del globals()[loaded_name]

gc.collect()
torch.cuda.empty_cache()


def load_indictrans_tokenizer(model_id, revision, cache_dir):
    tokenizer_class = get_class_from_dynamic_module(
        "tokenization_indictrans.IndicTransTokenizer",
        model_id,
        revision=revision,
        code_revision=revision,
        cache_dir=cache_dir,
    )
    return tokenizer_class.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
    )


print("Loading Whisper large-v3...")
ASR_MODEL = whisper.load_model(
    ASR_MODEL_ID,
    device="cuda",
    download_root=str(MODEL_ROOT / "whisper"),
).eval()

print("Loading NVIDIA NeMo Sortformer...")
DIARIZATION_MODEL = SortformerEncLabelModel.from_pretrained(
    DIARIZATION_MODEL_ID,
).to("cuda").eval()

print("Loading English → Hindi translation...")
en_hi_cache = str(MODEL_ROOT / "indictrans2_en_hi")
EN_HI_TOKENIZER = load_indictrans_tokenizer(
    EN_HI_MODEL_ID, EN_HI_REVISION, en_hi_cache
)
EN_HI_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    EN_HI_MODEL_ID,
    revision=EN_HI_REVISION,
    trust_remote_code=True,
    code_revision=EN_HI_REVISION,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=en_hi_cache,
).to("cuda").eval()

print("Loading Hindi → English translation...")
hi_en_cache = str(MODEL_ROOT / "indictrans2_hi_en")
HI_EN_TOKENIZER = load_indictrans_tokenizer(
    HI_EN_MODEL_ID, HI_EN_REVISION, hi_en_cache
)
HI_EN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    HI_EN_MODEL_ID,
    revision=HI_EN_REVISION,
    trust_remote_code=True,
    code_revision=HI_EN_REVISION,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=hi_en_cache,
).to("cuda").eval()

INDIC_PROCESSOR = IndicProcessor(inference=True)

print("Warming up Whisper...")
with INFERENCE_LOCK:
    ASR_MODEL.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        language="en",
        task="transcribe",
        verbose=None,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        fp16=True,
    )

print("All models are loaded once and ready.")

# ## Step 5 — Inspect channels and prepare purpose-specific 16 kHz paths
# 
# Audio Probing & Media Preparationresolve_media_path and probe_media: Extract and inspect the input audio/video metadata via ffprobe to check channels, duration, and sample rates.  
# run_ffmpeg: Standardizes audio into 16 kHz PCM WAVs, applying profile-specific filters for raw ASR, diarization filtering, or adaptive noise/dynamic compression enhancement.
# waveform_metrics: Analyzes audio signals using soundfile to calculate RMS dBFS, peak levels, clipping fraction, noise floor, and approximate SNR.  
# prepare_media: Directs stereo inputs into isolated left/right channel tracks or mono inputs into raw, enhanced, and diarization paths.


def normalize_spacing(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?।])", r"\1", text)

def resolve_media_path(media_value):
    candidate = media_value
    if isinstance(candidate, dict):
        candidate = candidate.get("path") or candidate.get("video") or candidate.get("name")
        if isinstance(candidate, dict):
            candidate = candidate.get("path") or candidate.get("name")
    elif isinstance(candidate, (tuple, list)):
        candidate = candidate[0] if candidate else None

    if hasattr(candidate, "video"):
        candidate = candidate.video
    if hasattr(candidate, "path"):
        candidate = candidate.path

    if not candidate:
        raise gr.Error("Record or upload a completed audio/video file first.")

    path = Path(str(candidate))
    if not path.exists() or not path.is_file():
        raise gr.Error("The media file is no longer available. Record or upload it again.")
    return path

def probe_media(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,channel_layout,sample_rate,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise gr.Error("This file does not contain a readable audio track.")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise gr.Error("This file does not contain an audio track.")

    stream = streams[0]
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)

    return {
        "codec": stream.get("codec_name"),
        "channels": int(stream.get("channels") or 1),
        "channel_layout": stream.get("channel_layout") or "unknown",
        "source_sample_rate": int(stream.get("sample_rate") or 0),
        "duration_seconds": duration,
    }

def run_ffmpeg(source, output, pan=None, enhanced=False, diarization=False):
    if enhanced and diarization:
        raise ValueError("ASR enhancement and diarization preparation are separate profiles.")

    filters = []
    expected_duration = None
    if pan is not None:
        filters.append(f"pan=mono|c0={pan}")

    filters.append(f"aresample={SAMPLE_RATE}")

    if enhanced:
        expected_duration = probe_media(source)["duration_seconds"]
        filters.extend([
            "highpass=f=65",
            "lowpass=f=7600",
            "afftdn=nf=-28:tn=1:tr=1",
            "acompressor=threshold=0.10:ratio=2:attack=20:release=250:makeup=1.35",
            "dynaudnorm=f=250:g=5:p=0.95:m=4",
            "alimiter=limit=0.95",
            "apad=pad_dur=0.10",
            f"atrim=duration={expected_duration:.6f}",
            "asetpts=N/SR/TB",
        ])
    elif diarization:
        filters.extend([
            "highpass=f=70",
            "lowpass=f=7600",
        ])

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-map", "0:a:0", "-vn",
        "-af", ",".join(filters), "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.exists():
        details = normalize_spacing(result.stderr or result.stdout)[-600:]
        raise gr.Error(f"FFmpeg could not prepare this audio: {details}")

def waveform_metrics(path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if sample_rate != SAMPLE_RATE or audio.size == 0:
        raise RuntimeError("Prepared WAV validation failed.")

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)))

    frame_size = max(1, int(0.10 * sample_rate))
    frame_count = len(audio) // frame_size

    if frame_count:
        frames = audio[:frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
        noise_floor_dbfs = float(np.percentile(frame_db, 20))
        speech_level_dbfs = float(np.percentile(frame_db, 90))
    else:
        noise_floor_dbfs = speech_level_dbfs = 20.0 * math.log10(max(rms, 1e-8))

    return {
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-8)), 2),
        "peak": round(peak, 4),
        "dc_offset": round(float(np.mean(audio)), 6),
        "clipping_fraction": round(float(np.mean(np.abs(audio) >= 0.999)), 6),
        "silence_fraction": round(float(np.mean(np.abs(audio) < 0.002)), 4),
        "approx_noise_floor_dbfs": round(noise_floor_dbfs, 2),
        "approx_speech_level_dbfs": round(speech_level_dbfs, 2),
        "approx_snr_db": round(max(0.0, speech_level_dbfs - noise_floor_dbfs), 2),
    }, audio

def prepare_media(media_value):
    source = resolve_media_path(media_value)
    metadata = probe_media(source)

    job_id = time.strftime("call-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = OUTPUT_ROOT / job_id
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=False)

    tracks = []

    # # === FAST STEREO METHOD ===
    # # Blindly splits 2-channel audio into Agent/Customer channels.
    # # Skips the slow correlation mathematics.
    # if metadata["channels"] == 2:
    #     left_source = audio_dir / "channel_1_source.wav"
    #     right_source = audio_dir / "channel_2_source.wav"

    #     run_ffmpeg(source, left_source, pan="c0", enhanced=False)
    #     run_ffmpeg(source, right_source, pan="c1", enhanced=False)

    #     left_metrics, left_audio = waveform_metrics(left_source)
    #     right_metrics, _ = waveform_metrics(right_source)

    #     # WEBM DURATION BUG FIX
    #     if metadata["duration_seconds"] == 0.0:
    #         metadata["duration_seconds"] = len(left_audio) / SAMPLE_RATE

    #     if metadata["duration_seconds"] < MIN_MEDIA_SECONDS:
    #         raise gr.Error("The recording is too short. Use at least 0.5 seconds.")

    #     tracks.append({
    #         "speaker": "Agent (Channel 1)",
    #         "raw_path": left_source,
    #         "enhanced_path": audio_dir / "channel_1_enhanced.wav",
    #         "raw_metrics": left_metrics,
    #     })
    #     tracks.append({
    #         "speaker": "Customer (Channel 2)",
    #         "raw_path": right_source,
    #         "enhanced_path": audio_dir / "channel_2_enhanced.wav",
    #         "raw_metrics": right_metrics,
    #     })
    #     metadata["channel_strategy"] = "separated_call_channels"

    # === FAST MONO METHOD ===
    # else:
    #     mono_raw = audio_dir / "mono_raw.wav"
    #     run_ffmpeg(source, mono_raw, pan=None, enhanced=False)

    #     metrics, mono_audio = waveform_metrics(mono_raw)

    #     # WEBM DURATION BUG FIX
    #     if metadata["duration_seconds"] == 0.0:
    #         metadata["duration_seconds"] = len(mono_audio) / SAMPLE_RATE

    #     if metadata["duration_seconds"] < MIN_MEDIA_SECONDS:
    #         raise gr.Error("The recording is too short. Use at least 0.5 seconds.")

    #     mono_enhanced = audio_dir / "mono_enhanced.wav"
    #     tracks.append({
    #         "speaker": None,
    #         "raw_path": mono_raw,
    #         "enhanced_path": mono_enhanced,
    #         "raw_metrics": metrics,
    #         "diarization_path": mono_raw,
    #         "diarization_metrics": metrics,
    #     })
    #     metadata["channel_strategy"] = "mono_diarization"


    # === FAST MONO METHOD ===
    # else:
    mono_raw = audio_dir / "mono_raw.wav"
    run_ffmpeg(source, mono_raw, pan=None, enhanced=False)
        
    metrics, mono_audio = waveform_metrics(mono_raw)
        
        # WEBM DURATION BUG FIX
    if metadata["duration_seconds"] == 0.0:
        metadata["duration_seconds"] = len(mono_audio) / SAMPLE_RATE
            
    if metadata["duration_seconds"] < MIN_MEDIA_SECONDS:
        raise gr.Error("The recording is too short. Use at least 0.5 seconds.")

    mono_enhanced = audio_dir / "mono_enhanced.wav"
    tracks.append({
            "speaker": None, 
            "raw_path": mono_raw,
            "enhanced_path": mono_enhanced, 
            "raw_metrics": metrics,
            "diarization_path": mono_raw,  # Default to raw audio
            "diarization_metrics": metrics,
    })
    metadata["channel_strategy"] = "mono_diarization"
    metadata["source_file"] = source.name
    return source, job_dir, tracks, metadata

# ## Step 6 — Run adaptive whole-file Whisper large-v3 decoding
# 
# Balanced mode starts with one raw/auto pass. Noise, sparse call legs, low confidence, or Hindi-English code-switching trigger targeted enhanced/language retries. Maximum-accuracy mode explicitly runs all six combinations. No application-level audio chunks are created.


import re
import numpy as np
import gradio as gr
from pathlib import Path
from flashtext import KeywordProcessor
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
# from flashtext import KeywordProcessor
# ------------------------------------------------------------------------------
# UTILITY & EVALUATION FUNCTIONS
# ------------------------------------------------------------------------------

def repeated_ngram_ratio(text, n=3):
    words = normalize_spacing(text).lower().split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return max(0.0, 1.0 - len(set(grams)) / max(1, len(grams)))

def contains_hinglish(text):
    devanagari = len(re.findall(r"[\u0900-\u097F]", str(text or "")))
    latin_words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{1,}\b", str(text or ""))
    return bool(devanagari >= 3 and latin_words)

def canonical_token(token):
    return re.sub(r"[^a-z0-9\u0900-\u097F]+", "", str(token or "").lower())

def collapse_consecutive_phrase_repeats(text):
    """Remove only obvious 3x phrase loops (or 4x one-word loops)."""
    tokens = normalize_spacing(text).split()
    if len(tokens) < 6:
        return normalize_spacing(text)
    output, index = [], 0
    while index < len(tokens):
        collapsed = False
        for size in range(min(12, (len(tokens) - index) // 3), 0, -1):
            required = 4 if size == 1 else 3
            if index + size * required > len(tokens):
                continue
            phrase = [canonical_token(value) for value in tokens[index:index + size]]
            if not all(phrase):
                continue
            repeats = 1
            while index + (repeats + 1) * size <= len(tokens):
                candidate = [
                    canonical_token(value)
                    for value in tokens[index + repeats * size:index + (repeats + 1) * size]
                ]
                if candidate != phrase:
                    break
                repeats += 1
            if repeats >= required:
                output.extend(tokens[index:index + size])
                index += repeats * size
                collapsed = True
                break
        if not collapsed:
            output.append(tokens[index])
            index += 1
    return normalize_spacing(" ".join(output))

def candidate_quality(candidate):
    text = candidate["text"]
    if not text:
        return -999.0
    segments = candidate["segments"]
    words = [word for segment in segments for word in segment.get("words", [])]
    mean_logprob = float(np.mean([segment["avg_logprob"] for segment in segments]))
    mean_word_probability = float(np.mean([word["probability"] for word in words])) if words else 0.0
    mean_no_speech = float(np.mean([segment["no_speech_prob"] for segment in segments]))
    compression_excess = float(np.mean([
        max(0.0, segment["compression_ratio"] - 2.2) for segment in segments
    ]))
    low_confidence_fraction = (
        float(np.mean([word["probability"] < 0.35 for word in words])) if words else 1.0
    )
    repetition = repeated_ngram_ratio(text)
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    letters = max(1, devanagari + latin + arabic)
    arabic_ratio = arabic / letters
    language_penalty = 0.0 if candidate["detected_language"] in {"en", "hi"} else 0.8
    requested = candidate["requested_language"]
    script_penalty = 0.0
    if requested == "hi" and devanagari == 0 and latin >= 20:
        script_penalty = 0.25
    elif requested == "en" and devanagari > 0:
        script_penalty = 0.50
    code_switch_bonus = 0.08 if requested == "hi" and devanagari and latin else 0.0
    return round(
        mean_logprob
        + 0.70 * mean_word_probability
        - 0.45 * mean_no_speech
        - 0.35 * compression_excess
        - 0.55 * low_confidence_fraction
        - 1.60 * repetition
        - 2.50 * arabic_ratio
        - language_penalty
        - script_penalty
        + code_switch_bonus,
        5,
    )

# def transcribe_candidate(
#     wav_path,
#     variant_name,
#     requested_language,
#     beam_size=ASR_BEAM_SIZE,
#     initial_prompt=None,
#     prompt_profile="none",
# ):  

#     safe_beam_size = max(2, int(beam_size))
#     print("\n" + "="*50)
#     print("=== [WHISPER INPUT] ===")
#     print(f"WAV Path: {wav_path}")
#     print(f"Variant: {variant_name}")
#     print(f"Requested Language: {requested_language or 'auto'}")
#     print(f"Beam Size: {beam_size}")
#     print(f"Initial Prompt: {initial_prompt}")
#     print(f"Prompt Profile: {prompt_profile}")
#     print("="*50)
    # result = ASR_MODEL.transcribe(
    #     str(wav_path),
    #     language=requested_language,
    #     task="transcribe",
    #     verbose=None,
    #     beam_size=int(beam_size),
    #     patience=1.2,
    #     temperature=(0.0, 0.2, 0.4),
    #     condition_on_previous_text=False,
    #     compression_ratio_threshold=2.4,
    #     logprob_threshold=-1.0,
    #     no_speech_threshold=0.60,
    #     word_timestamps=True,
    #     hallucination_silence_threshold=2.0,
    #     prepend_punctuations="\"'“¿([{-",
    #     append_punctuations="\"'.。,，!！?？:：”)]}、",
    #     initial_prompt=initial_prompt,
    #     fp16=True,
    # )
     # In transcribe_candidate (Step 6)
    # result = ASR_MODEL.transcribe(
    #     str(wav_path),
    #     language=requested_language,
    #     task="transcribe",
    #     verbose=None,
    #     # FORCE BEAM SIZE >= 2 so word timestamps never fail
    #     beam_size=max(2, int(beam_size)), 
    #     patience=1.0,
    #     temperature=0.0,
    #     condition_on_previous_text=False,
    #     compression_ratio_threshold=2.2,
    #     logprob_threshold=-1.0,
    #     no_speech_threshold=0.60,
    #     word_timestamps=True, # This will now work properly
    #     hallucination_silence_threshold=1.5,
    #     prepend_punctuations="\"'“¿([{-",
    #     append_punctuations="\"'.。,，!！?？:：”)]}、",
    #     initial_prompt=initial_prompt,
    #     fp16=True,
    # )
    
    # detected_lang =result.get("language")
    # detected_text = result.get("text","")
    # # contains Hallucinated languages 
    # contains_unexpected_script = bool(re.search(r"[\u0600-\u06FF]",detected_text)) # r"[\u0600-\u06FF]" this is urdu and arbic language matching

    # if requested_language is None and (detected_lang not in {"en","hi"} or contains_unexpected_script):
    #     print(f"[GUARDRAIL] Invalid language '{detected_lang}' detected on noisy audio. Testing English vs Hindi ..")

    # # Run a quick pass for english - expiclitly check only the mention language 
    # res_en = ASR_MODEL.transcribe(
    #     str(wav_path),
    #     language="en",
    #     beam_size= 2,
    #     temperature=0.0,
    #     fp16=True,
    # )
    # res_hi = ASR_MODEL.transcribe(
    #     str(wav_path),
    #     language ="hi",
    #     beam_size = 2,
    #     temperature = 0.0,
    #     fp16 = True,
    # )

    # #Pick candidate with higher log_probability
    # avg_logprob_en = np.mean([s["avg_logprob"] for s in res_en.get("segments", [])]) if res_en.get("segments") else -999
    # avg_logprob_hi = np.mean([s["avg_logprob"] for s in res_en.get("segments", [])]) if res_hi.get("segments") else -999

    # result = res_en if avg_logprob_en >= avg_logprob_hi else res_hi
    
    
    # print("\n" + "-"*50)
    # print("=== [WHISPER RAW OUTPUT] ===")
    # print(f"Detected Language: {result.get('language')}")
    # print(f"Full Text: {result.get('text')}")
    # print(f"Segment Count: {len(result.get('segments', []))}")
    
    # if result.get("segments"):
    #     first_seg = result["segments"][0]
    #     print(f"Sample Segment [0]: Start={first_seg.get('start')}s, End={first_seg.get('end')}s, Text='{first_seg.get('text')}'")
    #     print(f"Sample Words [0]: {first_seg.get('words', [])[:3]}...")
    # print("-" * 50 + "\n")
    
    # segments = []
    # for segment in result.get("segments", []):
    #     text = normalize_spacing(segment.get("text"))
    #     if not text:
    #         continue
    #     words = []
    #     for word in segment.get("words", []) or []:
    #         word_text = normalize_spacing(word.get("word"))
    #         if word_text:
    #             words.append({
    #                 "start_seconds": round(float(word.get("start", segment["start"])), 3),
    #                 "end_seconds": round(float(word.get("end", segment["end"])), 3),
    #                 "text": word_text,
    #                 "probability": round(float(word.get("probability", 0.0)), 5),
    #             })
    #     segments.append({
    #         "start_seconds": round(float(segment["start"]), 3),
    #         "end_seconds": round(float(segment["end"]), 3),
    #         "text": text,
    #         "avg_logprob": round(float(segment.get("avg_logprob", -10.0)), 5),
    #         "no_speech_prob": round(float(segment.get("no_speech_prob", 1.0)), 5),
    #         "compression_ratio": round(float(segment.get("compression_ratio", 0.0)), 5),
    #         "words": words,
    #     })
    # candidate = {
    #     "variant": variant_name,
    #     "requested_language": requested_language or "auto",
    #     "beam_size": int(beam_size),
    #     "prompt_profile": prompt_profile,
    #     "detected_language": result.get("language"),
    #     "language_probability": None,
    #     "segments": segments,
    #     "text": normalize_spacing(" ".join(item["text"] for item in segments)),
    # }
    # candidate["quality_score"] = candidate_quality(candidate)
    # return candidate
def transcribe_candidate(
    wav_path,
    variant_name,
    requested_language,
    beam_size=ASR_BEAM_SIZE,
    initial_prompt=None,
    prompt_profile="none",
):  
    # FORCE minimum beam size of 2 so word timestamps never fail
    safe_beam_size = max(2, int(beam_size))
    
    # FIX: Properly print the inputs (NOT the result, which doesn't exist yet)
    print("\n" + "=" * 50)
    print("=== [WHISPER INPUT] ===")
    print(f"WAV Path: {wav_path}")
    print(f"Variant: {variant_name}")
    print(f"Requested Language: {requested_language or 'auto'}")
    print(f"Beam Size: {safe_beam_size}")
    print("=" * 50)

    result = ASR_MODEL.transcribe(
        str(wav_path),
        language=requested_language,
        task="transcribe",
        verbose=None,
        beam_size=safe_beam_size,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.2,
        logprob_threshold=-1.0,
        no_speech_threshold=0.60,
        word_timestamps=True,
        hallucination_silence_threshold=1.5,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、",
        initial_prompt=initial_prompt,
        fp16=True,
    )
    
    # FIX: Added proper newlines to prevent syntax errors
    detected_lang = result.get("language")
    detected_text = result.get("text", "")
    contains_unexpected_script = bool(re.search(r"[\u0600-\u06FF]", detected_text))

    if requested_language is None and (detected_lang not in {"en", "hi"} or contains_unexpected_script):
        print(f"[GUARDRAIL] Invalid language '{detected_lang}' detected. Testing English vs Hindi...")
        
        # Ensure fallback passes also use beam_size=2 and word_timestamps=True
        res_en = ASR_MODEL.transcribe(str(wav_path), language="en", beam_size=2, temperature=0.0, word_timestamps=True, fp16=True)
        res_hi = ASR_MODEL.transcribe(str(wav_path), language="hi", beam_size=2, temperature=0.0, word_timestamps=True, fp16=True)

        avg_logprob_en = np.mean([s["avg_logprob"] for s in res_en.get("segments", [])]) if res_en.get("segments") else -999
        avg_logprob_hi = np.mean([s["avg_logprob"] for s in res_hi.get("segments", [])]) if res_hi.get("segments") else -999

        result = res_en if avg_logprob_en >= avg_logprob_hi else res_hi

    print("\n" + "-"*50)
    print("=== [WHISPER RAW OUTPUT] ===")
    print(f"Detected Language: {result.get('language')}")
    print(f"Full Text: {result.get('text')}")
    print(f"Segment Count: {len(result.get('segments', []))}")
    
    if result.get("segments"):
        first_seg = result["segments"][0]
        print(f"Sample Segment [0]: Start={first_seg.get('start')}s, End={first_seg.get('end')}s, Text='{first_seg.get('text')}'")
        print(f"Sample Words [0]: {first_seg.get('words', [])[:3]}...")
    print("-" * 50 + "\n")
    
    # FIX: Added proper newline here
    segments = []
    for segment in result.get("segments", []):
        text = normalize_spacing(segment.get("text"))
        if not text:
            continue
        words = []
        for word in segment.get("words", []) or []:
            word_text = normalize_spacing(word.get("word"))
            if word_text:
                words.append({
                    "start_seconds": round(float(word.get("start", segment["start"])), 3),
                    "end_seconds": round(float(word.get("end", segment["end"])), 3),
                    "text": word_text,
                    "probability": round(float(word.get("probability", 0.0)), 5),
                })
        segments.append({
            "start_seconds": round(float(segment["start"]), 3),
            "end_seconds": round(float(segment["end"]), 3),
            "text": text,
            "avg_logprob": round(float(segment.get("avg_logprob", -10.0)), 5),
            "no_speech_prob": round(float(segment.get("no_speech_prob", 1.0)), 5),
            "compression_ratio": round(float(segment.get("compression_ratio", 0.0)), 5),
            "words": words,
        })

    candidate = {
        "variant": variant_name,
        "requested_language": requested_language or "auto",
        "beam_size": safe_beam_size,
        "prompt_profile": prompt_profile,
        "detected_language": result.get("language"),
        "language_probability": None,
        "segments": segments,
        "text": normalize_spacing(" ".join(item["text"] for item in segments)),
    }
    candidate["quality_score"] = candidate_quality(candidate)
    return candidate


def needs_forced_language_retry(candidate):
    text = candidate.get("text", "")
    arabic_script = bool(re.search(r"[\u0600-\u06FF]", text))
    return bool(
        candidate.get("detected_language") not in {"en", "hi"}
        or candidate.get("quality_score", -999.0) < 0.05
        or arabic_script
        or repeated_ngram_ratio(text) > 0.18
    )

# ------------------------------------------------------------------------------
# MASTER REGISTRY & FLASHTEXT TRIE INDEX BUILDER
# ------------------------------------------------------------------------------
@dataclass
class CanonicalTermConfig:
    canonical: str
    local_cues: Set[str]
    acoustic_aliases: Dict[str, float] = field(default_factory=dict)
    minimum_similarity: float = 0.74

@dataclass
class RegionalAmountConfig:
    amount_cues: Set[str]
    units_map: Dict[str, str]
    ambiguous_suffixes: Set[str]
    singular: str = "lakh"
    plural: str = "lakhs"

@dataclass
class MasterDomainConfig:
    domain_id: str
    minimum_evidence_score: int
    initial_prompt: str
    evidence_weights: Dict[str, int]
    subcategories: Dict[str, List[str]]
    canonical_terms: Dict[str, CanonicalTermConfig] = field(default_factory=dict)
    amount_config: Optional[RegionalAmountConfig] = None

# MASTER SINGLE SOURCE OF TRUTH REGISTRY
MASTER_CALL_REGISTRY: Dict[str, MasterDomainConfig] = {
    "personal_loans": MasterDomainConfig(
        domain_id="personal_loans",
        minimum_evidence_score=5,
        initial_prompt="Indian English call regarding personal loans, EMI, KYC, PAN, salary, eligibility, and disbursement.",
        evidence_weights={
            "personal loan": 4, "loan amount": 3, "loan offer": 3,
            "eligibility": 2, "emi": 2, "kyc": 2, "salaried": 1, "self-employed": 1
        },
        subcategories={
            "application": ["apply", "eligibility", "process", "new loan", "documents required"],
            "emi_inquiry": ["schedule", "interest rate", "monthly emi", "due date"],
            "foreclosure": ["preclose", "foreclosure", "settlement"]
        },
        canonical_terms={
            "loan": CanonicalTermConfig(
                canonical="loan",
                acoustic_aliases={"look": 0.86, "lone": 0.94, "loam": 0.82},
                local_cues={"purpose", "amount", "offer", "eligibility", "application", "apply", "personal"}
            )
        },
        amount_config=RegionalAmountConfig(
            amount_cues={"amount", "how much", "loan", "borrow", "looking for", "required"},
            units_map={"lac": "lakh", "lacs": "lakhs", "cr": "crore"},
            ambiguous_suffixes={"x", "×"}
        )
    ),
    "credit_cards": MasterDomainConfig(
        domain_id="credit_cards",
        minimum_evidence_score=5,
        initial_prompt="Credit card call covering credit limit, annual fee, reward points, billing cycle, statement.",
        evidence_weights={
            "credit card": 4, "credit limit": 3, "annual fee": 3, "reward points": 2, "billing cycle": 2
        },
        subcategories={
            "activation": ["pin generation", "activate", "dispatch"],
            "limit_enhancement": ["limit increase", "enhancement", "upgrade"],
            "fee_waiver": ["annual fee", "charges reversal", "waive"]
        },
        canonical_terms={
            "card": CanonicalTermConfig(
                canonical="card",
                acoustic_aliases={"car": 0.85, "guard": 0.80, "cart": 0.88},
                local_cues={"credit", "debit", "limit", "pin", "statement"}
            )
        }
    ),
    "insurance": MasterDomainConfig(
        domain_id="insurance",
        minimum_evidence_score=5,
        initial_prompt="Insurance call regarding policy renewal, premium payment, claim settlement, health cover, and nominee.",
        evidence_weights={
            "policy": 4, "premium": 3, "insurance": 3, "claim": 3, "renewal": 2, "sum assured": 2
        },
        subcategories={
            "renewal": ["pay premium", "due date", "grace period", "lapsed policy"],
            "claims": ["hospitalization", "cashless", "reimbursement", "claim status"],
            "policy_details": ["nominee", "add-on", "coverage", "terms and conditions"]
        },
        canonical_terms={
            "policy": CanonicalTermConfig(
                canonical="policy",
                acoustic_aliases={"police": 0.85, "poly": 0.80},
                local_cues={"number", "details", "document", "insurance", "renewal", "claim"}
            ),
            "premium": CanonicalTermConfig(
                canonical="premium",
                acoustic_aliases={"premise": 0.78, "prepare": 0.75},
                local_cues={"pay", "amount", "due", "yearly", "monthly"}
            )
        }
    ),
    "debt_collections": MasterDomainConfig(
        domain_id="debt_collections",
        minimum_evidence_score=5,
        initial_prompt="Collections call regarding overdue payment, outstanding balance, settlement offer, payment deadline, and EMI bounce.",
        evidence_weights={
            "overdue": 4, "outstanding": 3, "settlement": 3, "payment": 3, "due date": 2, "bounce": 2
        },
        subcategories={
            "payment_promise": ["will pay", "pay tomorrow", "upi transfer", "net banking"],
            "dispute": ["already paid", "wrong amount", "financial crisis", "receipt"],
            "settlement": ["one time settlement", "waiver", "discounted close"]
        },
        canonical_terms={
            "dues": CanonicalTermConfig(
                canonical="dues",
                acoustic_aliases={"do": 0.80, "two": 0.75, "dew": 0.90},
                local_cues={"clear", "pending", "pay", "outstanding", "total"}
            )
        },
        amount_config=RegionalAmountConfig(
            amount_cues={"overdue", "outstanding", "pay", "settle", "due amount"},
            units_map={"lac": "lakh", "lacs": "lakhs", "cr": "crore"},
            ambiguous_suffixes={"x", "×"}
        )
    ),
    "ecommerce_delivery": MasterDomainConfig(
        domain_id="ecommerce_delivery",
        minimum_evidence_score=4,
        initial_prompt="E-commerce customer service call regarding order tracking, refund, return pickup, delivery status, and replacement.",
        evidence_weights={
            "order": 3, "delivery": 3, "refund": 3, "return": 2, "replacement": 2, "tracking": 2
        },
        subcategories={
            "tracking": ["where is my order", "out for delivery", "delayed", "shipment"],
            "returns": ["pickup agent", "refund status", "damaged item", "wrong size"]
        },
        canonical_terms={
            "order": CanonicalTermConfig(
                canonical="order",
                acoustic_aliases={"other": 0.76, "older": 0.80},
                local_cues={"track", "placed", "status", "id", "number", "delivery"}
            )
        }
    ),
    "utilities_broadband": MasterDomainConfig(
        domain_id="utilities_broadband",
        minimum_evidence_score=4,
        initial_prompt="Technical support call for broadband, router, wifi speed, electricity bill, power outage, and connection issue.",
        evidence_weights={
            "internet": 3, "broadband": 3, "router": 3, "wifi": 2, "not working": 2, "outage": 2
        },
        subcategories={
            "wifi_issue": ["router", "red light", "no signal", "wifi disconnected"],
            "speed_issue": ["slow speed", "buffering", "latency", "mbps"],
            "billing": ["high bill", "payment failed", "due date", "tariff plan"]
        },
        canonical_terms={
            "router": CanonicalTermConfig(
                canonical="router",
                acoustic_aliases={"route": 0.85, "daughter": 0.78, "rooter": 0.92},
                local_cues={"wifi", "red light", "restart", "box", "connection"}
            )
        }
    ),
    "general_call_center": MasterDomainConfig(
        domain_id="general_call_center",
        minimum_evidence_score=4,
        initial_prompt="Hello, am I speaking to Mr. Singh? Good morning sir. Myself Anuja calling from L&T Finance. Is it the right time to speak? Please note down one number. I will call you after 11.",
        evidence_weights={
            "am i speaking": 3, 
            "right time to speak": 4, 
            "calling from": 2, 
            "note down": 3,
            "turning down": 2, # Catch the common misheard variant as evidence
            "repeat": 2,
            "landline": 2,
            "thank you so much": 2
        },
        subcategories={
            "verification": ["am i speaking to", "is this", "calling from", "myself"],
            "rescheduling": ["right time", "call me after", "busy right now", "o'clock"],
            "information_exchange": ["note down", "number", "repeat", "landline"]
        },
        canonical_terms={
            "note down": CanonicalTermConfig(
                canonical="note down",
                acoustic_aliases={
                    "turning down": 0.88, 
                    "running down": 0.82,
                    "putting down": 0.80
                },
                # Only change "turning down" to "note down" if words like "number" or "one" are nearby
                local_cues={"number", "one", "a", "please", "my"} 
            ),
            "call you": CanonicalTermConfig(
                canonical="call you",
                acoustic_aliases={
                    "tell you": 0.85, 
                    "tell him": 0.80
                },
                # Only change "tell you" to "call you" if talking about time scheduling
                local_cues={"after", "o'clock", "later", "tomorrow", "back", "will"}
            ),
            "L&T Finance": CanonicalTermConfig(
                canonical="L&T Finance",
                acoustic_aliases={
                    "l &t finance": 0.95, 
                    "lnt finance": 0.90, 
                    "ellen t finance": 0.88,
                    "l and t finance": 0.92
                },
                local_cues={"from", "calling", "myself", "bank"}
            ),
            "landline": CanonicalTermConfig(
                canonical="landline",
                acoustic_aliases={
                    "land line": 0.95,
                    "online": 0.75,
                    "line": 0.70
                },
                local_cues={"number", "in", "on", "call"}
            )
        }
    )
}

def build_unified_flashtext_indices(registry: Dict[str, MasterDomainConfig]):
    tax_kp = KeywordProcessor(case_sensitive=False)
    corr_kp = KeywordProcessor(case_sensitive=False)

    for domain_id, config in registry.items():
        for term, weight in config.evidence_weights.items():
            tax_kp.add_keyword(term, {"type": "evidence", "category": domain_id, "weight": weight})
        for sub_id, keywords in config.subcategories.items():
            for kw in keywords:
                tax_kp.add_keyword(kw, {"type": "subcategory", "category": domain_id, "subcategory": sub_id})

        for _, term_cfg in config.canonical_terms.items():
            for alias in term_cfg.acoustic_aliases.keys():
                corr_kp.add_keyword(alias, {
                    "domain_id": domain_id,
                    "canonical": term_cfg.canonical,
                    "local_cues": term_cfg.local_cues,
                    "similarity": term_cfg.acoustic_aliases[alias]
                })

    return tax_kp, corr_kp

TAXONOMY_INDEX, CORRECTION_INDEX = build_unified_flashtext_indices(MASTER_CALL_REGISTRY) 

# CALL_TAXONOMY_REGISTRY = {
#     "personal_loans": {
#         "minimum_evidence_score": 5,
#         "initial_prompt": "Indian English call regarding personal loans, EMI, KYC, PAN, salary, eligibility, and disbursement.",
#         "evidence_weights": {
#             "personal loan": 4,
#             "loan amount": 3,
#             "loan offer": 3,
#             "eligibility": 2,
#             "emi": 2,
#             "kyc": 2,
#             "disbursement": 2,
#         },
#         "subcategories": {
#             "application": ["apply", "eligibility", "process", "new loan", "documents required"],
#             "emi_inquiry": ["schedule", "interest rate", "monthly emi", "due date", "bouncing charges"],
#             "foreclosure": ["preclose", "foreclosure", "settlement", "closure letter"],
#         },
#         "aliases": {"look": "loan", "lone": "loan", "loam": "loan", "lon": "loan"},
#     },
#     "credit_cards": {
#         "minimum_evidence_score": 5,
#         "initial_prompt": "Credit card call covering credit limit, annual fee, reward points, billing cycle, statement, and block card.",
#         "evidence_weights": {
#             "credit card": 4,
#             "credit limit": 3,
#             "annual fee": 3,
#             "reward points": 2,
#             "billing cycle": 2,
#             "statement": 2,
#         },
#         "subcategories": {
#             "activation": ["pin generation", "activate", "dispatch", "delivery status"],
#             "limit_enhancement": ["limit increase", "enhancement", "upgrade", "credit score"],
#             "fee_waiver": ["annual fee", "charges reversal", "waive", "hidden charges"],
#             "lost_stolen": ["block card", "lost", "fraudulent transaction", "dispute"],
#         },
#         "aliases": {"car": "card", "guard": "card", "cart": "card", "cat": "card"},
#     },
    # "insurance": {
    #     "minimum_evidence_score": 5,
    #     "initial_prompt": "Insurance call regarding policy renewal, premium payment, claim settlement, health cover, and nominee.",
    #     "evidence_weights": {
    #         "policy": 4,
    #         "premium": 3,
    #         "insurance": 3,
    #         "claim": 3,
    #         "renewal": 2,
    #         "sum assured": 2,
    #     },
    #     "subcategories": {
    #         "renewal": ["pay premium", "due date", "grace period", "lapsed policy"],
    #         "claims": ["hospitalization", "cashless", "reimbursement", "claim status"],
    #         "policy_details": ["nominee", "add-on", "coverage", "terms and conditions"],
    #     },
    #     "aliases": {"police": "policy", "poly": "policy", "premium": "premium"},
    # },
    # "tech_support": {
    #     "minimum_evidence_score": 4,
    #     "initial_prompt": "Technical support call for broadband, router, wifi speed, outage, connection issue, and fiber link.",
    #     "evidence_weights": {
    #         "internet": 3,
    #         "broadband": 3,
    #         "router": 3,
    #         "wifi": 2,
    #         "not working": 2,
    #         "outage": 2,
    #     },
    #     "subcategories": {
    #         "wifi_issue": ["router", "red light", "no signal", "wifi disconnected", "restart"],
    #         "speed_issue": ["slow speed", "buffering", "latency", "mbps speed test"],
    #         "billing_plan": ["plan upgrade", "data quota", "fup limit"],
    #     },
    #     "aliases": {"route": "router", "daughter": "router", "ワイファイ": "wifi"},
    # },
    # "banking_general": {
    #     "minimum_evidence_score": 4,
    #     "initial_prompt": "General banking call for savings account, minimum balance, net banking password, debit card, and UPI.",
    #     "evidence_weights": {
    #         "savings account": 3,
    #         "net banking": 3,
    #         "debit card": 3,
    #         "minimum balance": 2,
    #         "upi": 2,
    #     },
    #     "subcategories": {
    #         "credentials": ["reset password", "user id locked", "mpin", "otp issue"],
    #         "account_status": ["min balance charge", "statement download", "branch address"],
    #     },
    #     "aliases": {"balance": "balance", "account": "account"},
    # },
    # "ecommerce_delivery": {
    #     "minimum_evidence_score": 4,
    #     "init_prompt": "E-commerce customer service call regarding order tracking, refund, return pickup, and replacement.",
    #     "evidence_weights": {
    #         "order": 3,
    #         "delivery": 3,
    #         "refund": 3,
    #         "return": 2,
    #         "replacement": 2,
    #     },
    #     "subcategories": {
    #         "tracking": ["where is my order", "out for delivery", "delayed"],
    #         "returns": ["pickup agent", "refund status", "damaged item", "wrong size"],
    #     },
    #     "aliases": {"order": "order", "delivery": "delivery"},
    # },
    # "insurance": {
    #     "minimum_evidence_score": 5,
    #     "initial_prompt": "Insurance call regarding policy renewal, premium payment, claim settlement, health cover, and nominee.",
    #     "evidence_weights": {
    #         "policy": 4,
    #         "premium": 3,
    #         "insurance": 3,
    #         "claim": 3,
    #         "renewal": 2,
    #         "sum assured": 2,
    #     },
    #     "subcategories": {
    #         "renewal": ["pay premium", "due date", "grace period", "lapsed policy"],
    #         "claims": ["hospitalization", "cashless", "reimbursement", "claim status"],
    #         "policy_details": ["nominee", "add-on", "coverage", "terms and conditions"],
    #     },
    #     "aliases": {"police": "policy", "poly": "policy", "premium": "premium"},
    # },
    # "debt_collection": {
    #     "minimum_evidence_score": 5,
    #     "initial_prompt": "Collections call regarding overdue payment, outstanding balance, settlement offer, payment deadline, and legal notice.",
    #     "evidence_weights": {
    #         "overdue": 4,
    #         "outstanding": 3,
    #         "settlement": 3,
    #         "payment": 3,
    #         "due date": 2,
    #         "installment": 2,
    #     },
    #     "subcategories": {
    #         "payment_promise": ["will pay", "pay tomorrow", "upi transfer", "net banking"],
    #         "dispute": ["already paid", "wrong amount", "harassment", "financial crisis"],
    #         "settlement": ["one time settlement", "waiver", "discounted close"],
    #     },
    #     "aliases": {"due": "due", "pay": "pay", "balance": "balance"},
    # },
    # "utilities_energy": {
    #     "minimum_evidence_score": 4,
    #     "initial_prompt": "Utility support call for electricity bill, power outage, gas connection, meter reading, and complaint.",
    #     "evidence_weights": {
    #         "electricity": 3,
    #         "power cut": 3,
    #         "meter": 3,
    #         "bill amount": 2,
    #         "outage": 2,
    #     },
    #     "subcategories": {
    #         "billing": ["high bill", "payment failed", "last date", "consumption"],
    #         "outage_complaint": ["no power", "transformer blast", "voltage fluctuation", "restoration time"],
    #     },
    #     "aliases": {"meter": "meter", "bill": "bill"},
    # },
#     "travel_hospitality": {
#         "minimum_evidence_score": 4,
#         "initial_prompt": "Travel support call for flight cancellation, ticket refund, hotel booking modification, and baggage delay.",
#         "evidence_weights": {
#             "flight": 3,
#             "booking": 3,
#             "cancellation": 3,
#             "refund": 3,
#             "hotel": 2,
#             "pnr": 2,
#         },
#         "subcategories": {
#             "cancellation": ["cancel ticket", "full refund", "flight delayed", "missed connection"],
#             "modification": ["change date", "seat upgrade", "extra baggage"],
#         },
#         "aliases": {"pn": "pnr", "ticket": "ticket"},
#     },
#     "healthcare_appointment": {
#         "minimum_evidence_score": 4,
#         "initial_prompt": "Healthcare call regarding doctor consultation, clinic appointment, prescription refill, and lab test reports.",
#         "evidence_weights": {
#             "appointment": 3,
#             "doctor": 3,
#             "clinic": 2,
#             "prescription": 2,
#             "report": 2,
#         },
#         "subcategories": {
#             "booking": ["schedule slot", "morning shift", "reschedule", "cancel visit"],
#             "reports": ["blood test", "x-ray", "consultation fee", "online prescription"],
#         },
#         "aliases": {"doc": "doctor", "app": "appointment"},
#     },
# }

# def build_flashtext_engine(registry):
#     """Builds a single O(N) Trie Search Index containing all categories, subcategories, and aliases."""
#     kp = KeywordProcessor(case_sensitive=False)
#     for cat_id, profile in registry.items():
#         # Add primary evidence keywords
#         for term, weight in profile["evidence_weights"].items():
#             kp.add_keyword(term, {"type": "evidence", "category": cat_id, "weight": weight})
        
#         # Add subcategory keywords
#         for sub_id, keywords in profile["subcategories"].items():
#             for kw in keywords:
#                 kp.add_keyword(kw, {"type": "subcategory", "category": cat_id, "subcategory": sub_id})

#         # Add acoustic aliases
#         for alias, canonical in profile.get("aliases", {}).items():
#             kp.add_keyword(alias, {"type": "alias", "category": cat_id, "canonical": canonical})
            
#     return kp

# # Initialize the global FlashText index ONCE at startup
# FLASHTEXT_INDEX = build_flashtext_engine(CALL_TAXONOMY_REGISTRY)


def classify_and_detect_anomalies_flashtext(text):
    clean_text = normalize_spacing(text).lower()
    matches = TAXONOMY_INDEX.extract_keywords(clean_text)

    category_scores = {}
    subcategory_scores = {}

    for meta in matches:
        m_type = meta["type"]
        cat_id = meta["category"]
        if m_type == "evidence":
            category_scores[cat_id] = category_scores.get(cat_id, 0) + meta["weight"]
        elif m_type == "subcategory":
            sub_key = (cat_id, meta["subcategory"])
            subcategory_scores[sub_key] = subcategory_scores.get(sub_key, 0) + 1

    valid_domains = [
        (score, cat_id) for cat_id, score in category_scores.items()
        if score >= MASTER_CALL_REGISTRY[cat_id].minimum_evidence_score
    ]

    if not valid_domains:
        return False, None

    best_score, best_cat_id = max(valid_domains, key=lambda x: x[0])
    best_profile = MASTER_CALL_REGISTRY[best_cat_id]

    best_sub_id = "general_inquiry"
    valid_subs = {sub: score for (c, sub), score in subcategory_scores.items() if c == best_cat_id}
    if valid_subs:
        best_sub_id = max(valid_subs, key=valid_subs.get)

    alias_matches = CORRECTION_INDEX.extract_keywords(clean_text)

    surface_anomaly = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*[x×]\b", clean_text)
        or re.search(r"(?i)\b([a-z']+)(?:\s+\1\b)+", clean_text)
        or len(alias_matches) > 0
    )

    domain_info = {
        "category": best_cat_id,
        "subcategory": best_sub_id,
        "score": best_score,
        "initial_prompt": best_profile.initial_prompt,
        "prompt_profile": f"{best_cat_id}-{best_sub_id}",
    }

    return surface_anomaly, domain_info



# def classify_and_detect_anomalies_flashtext(text):
#     """Extracts all matching taxonomy tokens in a SINGLE O(N) pass across the entire text."""
#     clean_text = normalize_spacing(text).lower()
#     matches = FLASHTEXT_INDEX.extract_keywords(clean_text)

#     category_scores = {}
#     subcategory_scores = {}
#     aliases_found = []

#     # Aggregate extracted match metadata instantly
#     for meta in matches:
#         m_type = meta["type"]
#         cat_id = meta["category"]

#         if m_type == "evidence":
#             category_scores[cat_id] = category_scores.get(cat_id, 0) + meta["weight"]
#         elif m_type == "subcategory":
#             sub_key = (cat_id, meta["subcategory"])
#             subcategory_scores[sub_key] = subcategory_scores.get(sub_key, 0) + 1
#         elif m_type == "alias":
#             aliases_found.append(meta)

#     # Filter categories meeting their evidence threshold
#     valid_domains = [
#         (score, cat_id) for cat_id, score in category_scores.items()
#         if score >= CALL_TAXONOMY_REGISTRY[cat_id]["minimum_evidence_score"]
#     ]

#     if not valid_domains:
#         return False, None

#     # Pick the winning category
#     best_score, best_cat_id = max(valid_domains, key=lambda x: x[0])
#     best_profile = CALL_TAXONOMY_REGISTRY[best_cat_id]

#     # Pick the winning subcategory
#     best_sub_id = "general_inquiry"
#     valid_subs = {sub: score for (c, sub), score in subcategory_scores.items() if c == best_cat_id}
#     if valid_subs:
#         best_sub_id = max(valid_subs, key=valid_subs.get)

    # Check for surface/acoustic anomalies triggering a retry
    # surface_anomaly = bool(
    #     re.search(r"\b\d+(?:\.\d+)?\s*[x×]\b", clean_text)
    #     or re.search(r"(?i)\b([a-z']+)(?:\s+\1\b)+", clean_text)
    #     or len(aliases_found) > 0
    # )

    # domain_info = {
    #     "category": best_cat_id,
    #     "subcategory": best_sub_id,
    #     "score": best_score,
    #     "initial_prompt": best_profile["initial_prompt"],
    #     "prompt_profile": f"{best_cat_id}-{best_sub_id}",
    # }

    # return surface_anomaly, domain_info

def needs_domain_context_retry(candidate):
    text = candidate.get("text", "")
    return classify_and_detect_anomalies_flashtext(text)

# ------------------------------------------------------------------------------
# MAIN PIPELINE ORCHESTRATOR
# ------------------------------------------------------------------------------

def transcribe_accuracy_first(
    track,
    processing_profile="Balanced (recommended)",
    progress=None,
    track_index=1,
    track_count=1,
):
    candidates = []
    paths = {"raw": track["raw_path"], "enhanced": track["enhanced_path"]}
    if track.get("crosstalk_path") is not None:
        paths["crosstalk"] = track["crosstalk_path"]
    maximum_accuracy = str(processing_profile).startswith("Maximum")

    def run_pass(
        variant,
        language,
        beam_size,
        description,
        initial_prompt=None,
        prompt_profile="none",
    ):
        if variant == "enhanced" and not Path(paths[variant]).exists():
            if progress:
                progress(
                    0.16 + 0.40 * track_index / max(1, track_count),
                    desc=f"Track {track_index}/{track_count}: preparing evidence-triggered enhanced audio...",
                )
            run_ffmpeg(paths["raw"], paths["enhanced"], enhanced=True)
        if progress:
            progress(
                0.18 + 0.40 * track_index / max(1, track_count),
                desc=(
                    f"Track {track_index}/{track_count}: {description} "
                    f"({variant}, {language or 'auto'}, beam {beam_size})"
                ),
            )
        candidate = transcribe_candidate(
            paths[variant],
            variant,
            language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            prompt_profile=prompt_profile,
        )
        candidates.append(candidate)
        return candidate

    with INFERENCE_LOCK:
        if maximum_accuracy:
            variants = ["raw"]
            if "crosstalk" in paths:
                variants.append("crosstalk")
            variants.append("enhanced")
            pass_total = len(variants) * 3
            for variant in variants:
                for language in (None,):
                    run_pass(
                        variant,
                        language,
                        ASR_RETRY_BEAM_SIZE,
                        f"maximum-accuracy pass {len(candidates) + 1}/{pass_total}",
                    )
        else:
            primary = run_pass(
                "raw", None, ASR_BEAM_SIZE, "primary whole-file Whisper pass"
            )

            if "crosstalk" in paths:
                run_pass(
                    "crosstalk", None, ASR_BEAM_SIZE,
                    "crosstalk-reduced comparison pass",
                )

            noisy_audio = float(track["raw_metrics"].get("approx_snr_db", 99.0)) < 12.0
            sparse_call_leg = float(track["raw_metrics"].get("silence_fraction", 0.0)) > 0.70
            usable_now = [item for item in candidates if item["segments"]]
            best_now = max(
                usable_now or candidates, key=lambda item: item["quality_score"]
            )
            if (
                noisy_audio
                or sparse_call_leg
                or contains_hinglish(best_now.get("text", ""))
                or best_now["quality_score"] < 0.20
            ):
                run_pass(
                    "enhanced", None, ASR_BEAM_SIZE,
                    "noise/sparse/multilingual enhanced retry",
                )
                usable_now = [item for item in candidates if item["segments"]]
                best_now = max(
                    usable_now or candidates, key=lambda item: item["quality_score"]
                )

        # Dynamic FlashText Taxonomy Classification & Retry
        usable_for_context = [item for item in candidates if item["segments"]]
        if usable_for_context:
            context_probe = max(
                usable_for_context, key=lambda item: item["quality_score"]
            )
            should_retry, domain_info = needs_domain_context_retry(context_probe)
            if should_retry and domain_info:
                prompt_language = (
                    "en" if context_probe.get("detected_language") == "en" else None
                )
                run_pass(
                    context_probe["variant"],
                    prompt_language,
                    ASR_RETRY_BEAM_SIZE,
                    f"flashtext retry ({domain_info['category']} -> {domain_info['subcategory']})",
                    initial_prompt=domain_info["initial_prompt"],
                    prompt_profile=domain_info["prompt_profile"],
                )

    usable = [candidate for candidate in candidates if candidate["segments"]]
    if not usable:
        raise gr.Error("Whisper did not find speech in this recording.")
    
    character_counts = [len(item["text"]) for item in usable]
    median_characters = max(1.0, float(np.median(character_counts)))
    for item in usable:
        coverage_ratio = len(item["text"]) / median_characters
        omission_penalty = max(0.0, 0.60 - coverage_ratio)
        item["selection_score"] = round(
            float(item["quality_score"]) - 0.35 * omission_penalty, 5
        )
    selected = max(usable, key=lambda item: item["selection_score"])
    summary = [
        {
            "variant": item["variant"],
            "requested_language": item["requested_language"],
            "detected_language": item["detected_language"],
            "language_probability": item["language_probability"],
            "beam_size": item["beam_size"],
            "prompt_profile": item.get("prompt_profile", "none"),
            "quality_score": item["quality_score"],
            "selection_score": item.get("selection_score", item["quality_score"]),
            "characters": len(item["text"]),
        }
        for item in sorted(
            candidates,
            key=lambda value: value.get("selection_score", value["quality_score"]),
            reverse=True,
        )
    ]
    return selected, summary

# ## Step 7 — Label speakers, preserve overlap and use later evidence
# 
# Separate call channels become stable Channel 1/2 labels. Mono recordings use NeMo Sortformer with batch size 1, NVIDIA's documented onset/offset post-processing, and the complete speaker-preserving waveform. NeMo labels are gated to recognized word-time regions without changing timestamps. A conservative second pass can revise an earlier uncertain non-overlap fragment when its acoustic candidate agrees with nearby confirmed speech, including a later turn. Conflicts and overlaps remain `Speaker uncertain`; the notebook does not claim to know a person's real identity.


NEMO_ACCURACY_CONFIG = {
    "chunk_len": 340,   # model is processing exactly 340 frames of audio at a time ~ approx 3.4 sec
    "chunk_right_context": 40,
    "fifo_len": 40, # fifo memory buffers
    "spkcache_update_period": 340, #how frequenty model refreshes its memory
    "spkcache_len": 188, # how much historical audio the model holds onto to define what speaker1 - sound like
}

SORTFORMER_POSTPROCESS_YAML = MODEL_ROOT / "sortformer_callcenter_postprocess.yaml"
SORTFORMER_POSTPROCESS_YAML.write_text(
#     """parameters:
#   onset: 0.64  # Lowering from 0.64 makes speaker detection start earlier
#   offset: 0.74 # Lowering from 0.74 holds the speaker identity longer over pauses
#   pad_onset: 0.06
#   pad_offset: 0.0
#   min_duration_on: 0.10
#   min_duration_off: 0.15
# """,
      """parameters:
  onset: 0.40   
  offset: 0.50  
  pad_offset: 0.05
  min_duration_on: 0.05
  min_duration_off: 0.10
""",
    encoding="utf-8",
) # diarization model outputs raw probabilities onset: 0.64: model must be at least 64% confident that human speech is occurring before it triggers a "Speaker Start" timestamp.
#offset: 0.74: The stopping threshold. The model will declare a speaker has stopped talking when confidence drops below 74%
# pad_onset: 0.06: Adds 60 milliseconds to the beginning of a detected speech block
# pad_offset: 0.0: Adds time to the end of a block. It is set to zero here because NeMo's acoustic model tends to naturally trail off, so no artificial padding is needed.
# min_duration_off: 0.15 : minimum allowed gap between words


def configure_nemo_accuracy():
    modules = DIARIZATION_MODEL.sortformer_modules
    for name, value in NEMO_ACCURACY_CONFIG.items():
        setattr(modules, name, int(value))
    if hasattr(DIARIZATION_MODEL, "_check_streaming_parameters"):
        DIARIZATION_MODEL._check_streaming_parameters()
    elif hasattr(modules, "_check_streaming_parameters"):
        modules._check_streaming_parameters()


def normalize_speaker_name(raw_name):
    match = re.search(r"(\d+)$", str(raw_name))
    return f"Speaker {int(match.group(1)) + 1}" if match else str(raw_name)

# [Raw Speaker Timeline]
def parse_nemo_lines(lines): # acts as a translator and validator between raw output generated by NeMo speaker diarization model and rest pipeline
    timeline = []  # ["0.540 2.100 speaker_0", "2.150 4.300 speaker_1", ...]
    for line in lines or []:
        parts = str(line).strip().split()
        if len(parts) < 3:
            continue
        try:
            start, end = max(0.0, float(parts[0])), float(parts[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        timeline.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "speaker": normalize_speaker_name(parts[2]),
        })
    return timeline


def run_nemo_diarization(wav_path):
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError("NeMo input must be 16 kHz.")
    if len(audio) < 2 * SAMPLE_RATE:
        return []
    # SAMPLE_RATE = 16,000 , samples per second (16 kHz).  2 * SAMPLE_RATE = 2*times = 2*16,000 = 32,000 audio samples.
    # len(audio) = The total count of floating-point audio numbers in the array.
    # If len(audio) is less than 32,000, the file is under 2 seconds long, so the function skips processing and returns an empty list ([])    
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

      # ------------------ PRINT NEMO INPUTS ------------------
    print("\n" + "="*50)
    print("=== [NEMO DIARIZATION INPUT] ===")
    print(f"WAV Path: {wav_path}")
    print(f"Audio Array Shape: {audio.shape}")
    print(f"Audio Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Batch Size: 1")
    print(f"Post-processing YAML: {SORTFORMER_POSTPROCESS_YAML}")
    print("="*50)
    with INFERENCE_LOCK:  # Thread lock wrapper that prevents multiple requests from accessing the GPU simultaneously avoid cuda race conditions/OOM(out of memorey)
        configure_nemo_accuracy() # applies calbrated streaming params on 
        predicted = DIARIZATION_MODEL.diarize( 
            audio=[np.clip(audio, -1.0, 1.0)],
            sample_rate=SAMPLE_RATE,
            batch_size=1,
            postprocessing_yaml=str(SORTFORMER_POSTPROCESS_YAML),
            num_workers=0,
            verbose=False,
        )
    if isinstance(predicted, tuple):
        predicted = predicted[0]
    lines = predicted[0] if predicted and isinstance(predicted[0], list) else predicted
      # ------------------ PRINT NEMO OUTPUTS ------------------
    print("\n" + "-"*50)
    print("=== [NEMO DIARIZATION RAW OUTPUT] ===")
    print(f"Raw Output Object: {predicted}")
    
    # if isinstance(predicted, tuple):
    #     predicted = predicted[0]
    # lines = predicted[0] if predicted and isinstance(predicted[0], list) else predicted
    
    print(f"Parsed Speaker Timeline Lines Count: {len(lines)}")
    print("Sample Speaker Lines:")
    for line in lines[:5]:  # Print first 5 speaker lines
        print(f"  {line}")
    print("-" * 50 + "\n")
    return parse_nemo_lines(lines)


def speaker_overlap_scores(start, end, timeline):
    start, end = float(start), float(end)
    if end <= start:
        midpoint = start
        start, end = max(0.0, midpoint - 0.02), midpoint + 0.02
    duration = max(0.04, end - start)
    overlaps = {}
    for item in timeline:
        overlap = max(
            0.0,
            min(end, item["end_seconds"]) - max(start, item["start_seconds"]),
        )
        if overlap > 0:
            overlaps[item["speaker"]] = overlaps.get(item["speaker"], 0.0) + overlap
    return [
        {
            "speaker": speaker,
            "overlap_seconds": round(seconds, 4),
            "overlap_ratio": round(min(1.0, seconds / duration), 4),
        }
        for speaker, seconds in sorted(overlaps.items(), key=lambda pair: pair[1], reverse=True)
    ]


# def fallback_split_segment_words(segment):
#     """Generates proportional word timing when Whisper word_timestamps returns empty."""
#     text = normalize_spacing(segment.get("text", ""))
#     if not text:
#         return []
    
#     raw_words = text.split()
#     if not raw_words:
#         return []
    
#     start = float(segment.get("start_seconds", segment.get("start", 0.0)))
#     end = float(segment.get("end_seconds", segment.get("end", start + 1.0)))
#     duration = max(0.1, end - start)
#     total_chars = max(1, sum(len(w) for w in raw_words))
    
#     words = []
#     current_start = start
#     for w_text in raw_words:
#         w_dur = (len(w_text) / total_chars) * duration
#         w_end = current_start + w_dur
#         words.append({
#             "start_seconds": round(current_start, 3),
#             "end_seconds": round(w_end, 3),
#             "text": w_text,
#             "probability": float(segment.get("avg_logprob", 0.80)),
#         })
#         current_start = w_end
#     return words



# Flattens Whisper segments into word-level timing and probability units.
def candidate_words(candidate):
    units = []
    for segment in candidate["segments"]:
        if segment["words"]:
            units.extend(segment["words"])
        else:
            units.append({
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "text": segment["text"],
                "probability": max(0.0, min(1.0, math.exp(segment["avg_logprob"]))),
            })
    return units
    
# Combines adjacent or overlapping time intervals within a maximum gap threshold
def merge_intervals(intervals, maximum_gap=0.15):
    merged = []
    for start, end in sorted(intervals):
        start, end = max(0.0, float(start)), float(end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + maximum_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
# Creates padded time windows around confirmed spoken words
def recognized_speech_intervals(candidate, padding=0.20):
    intervals = [
        (
            max(0.0, float(word["start_seconds"]) - padding),
            float(word["end_seconds"]) + padding,
        )
        for word in candidate_words(candidate)
        if normalize_spacing(word.get("text"))
    ]
    return merge_intervals(intervals, maximum_gap=0.20)

def gate_timeline_to_recognized_speech(timeline, candidate):
    """Remove non-speech labels while retaining original recording timestamps."""
    speech_intervals = recognized_speech_intervals(candidate)
    gated = []
    for item in timeline:
        for speech_start, speech_end in speech_intervals:
            start = max(float(item["start_seconds"]), speech_start)
            end = min(float(item["end_seconds"]), speech_end)
            if end - start >= 0.04:
                gated.append({
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "speaker": item["speaker"],
                })
    # Merge per speaker so another overlapping speaker cannot interrupt the merge.
    by_speaker = {}
    for item in gated:
        by_speaker.setdefault(item["speaker"], []).append(
            (item["start_seconds"], item["end_seconds"])
        )
    merged = [
        {
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "speaker": speaker,
        }
        for speaker, intervals in by_speaker.items()
        for start, end in merge_intervals(intervals, maximum_gap=0.05)
    ]
    return sorted(merged, key=lambda row: (row["start_seconds"], row["speaker"])), speech_intervals

# Computes cumulative active speaking duration for each detected speaker
def timeline_speaker_durations(timeline):
    durations = {}
    for item in timeline:
        duration = max(0.0, item["end_seconds"] - item["start_seconds"])
        durations[item["speaker"]] = durations.get(item["speaker"], 0.0) + duration
    return {speaker: round(seconds, 3) for speaker, seconds in sorted(durations.items())}


def build_mono_speaker_turns(candidate, timeline):
    turns = []
    for word in candidate_words(candidate):
        scores = speaker_overlap_scores(
            word["start_seconds"], word["end_seconds"], timeline
        )
        primary_ratio = scores[0]["overlap_ratio"] if scores else 0.0
        # NeMo onset/offset post-processing can leave a short edge gap around a
        # real word. Use a small tolerance only for the primary label; overlap
        # detection below still uses the unpadded word interval.
        nearby_scores = speaker_overlap_scores(
            max(0.0, float(word["start_seconds"]) - 0.20),
            float(word["end_seconds"]) + 0.20,
            timeline,
        )
        active_scores = [
            score for score in scores
            if score["overlap_seconds"] >= 0.04 and score["overlap_ratio"] >= 0.12
        ]

        # Primary Speaker Assignment
        if scores and primary_ratio >= 0.30:
            primary = scores[0]["speaker"]
        elif nearby_scores and nearby_scores[0]["overlap_ratio"] >= 0.20:
            primary = nearby_scores[0]["speaker"]
        else:
            primary = "Speaker uncertain"
        overlap = len(active_scores) > 1
        close_second = bool(
            len(active_scores) > 1
            and active_scores[1]["overlap_ratio"]
            >= 0.60 * active_scores[0]["overlap_ratio"]
        )
        assignment_uncertain = bool(
            primary == "Speaker uncertain" or primary_ratio < 0.50 or close_second
        )
        active_speakers = [score["speaker"] for score in active_scores]
        if not active_speakers and scores:
            active_speakers = [scores[0]["speaker"]]
        if not active_speakers and primary != "Speaker uncertain":
            active_speakers = [primary]
        if not active_speakers:
            active_speakers = ["Speaker uncertain"]
        can_merge = bool(
            turns
            and turns[-1]["speaker"] == primary
            and turns[-1]["overlap_detected"] == overlap
            and turns[-1]["speaker_assignment_uncertain"] == assignment_uncertain
            and set(turns[-1]["active_speakers"]) == set(active_speakers)
            and word["start_seconds"] - turns[-1]["end_seconds"] <= 1.25
            and word["end_seconds"] - turns[-1]["start_seconds"] <= 20.0
        )
        if can_merge:
            turn = turns[-1]
            turn["end_seconds"] = word["end_seconds"]
            turn["transcript"] = normalize_spacing(turn["transcript"] + " " + word["text"])
            turn["active_speakers"] = sorted(
                set(turn["active_speakers"] + active_speakers)
            )
            turn["_probabilities"].append(float(word["probability"]))
            turn["_speaker_ratios"].append(float(primary_ratio))
            turn["_uncertain_flags"].append(float(assignment_uncertain))
        else:
            turns.append({
                "start_seconds": word["start_seconds"],
                "end_seconds": word["end_seconds"],
                "source_track_index": 0,
                "speaker": primary,
                "active_speakers": active_speakers,
                "overlap_detected": overlap,
                "speaker_assignment_uncertain": assignment_uncertain,
                "transcript": word["text"],
                "whisper_language": candidate.get("detected_language"),
                "whisper_requested_language": candidate.get("requested_language"),
                "_probabilities": [float(word["probability"])],
                "_speaker_ratios": [float(primary_ratio)],
                "_uncertain_flags": [float(assignment_uncertain)],
            })
    for turn in turns:
        turn["word_confidence"] = round(float(np.mean(turn.pop("_probabilities"))), 4)
        turn["speaker_overlap_ratio"] = round(
            float(np.mean(turn.pop("_speaker_ratios"))), 4
        )
        turn["speaker_uncertainty_fraction"] = round(
            float(np.mean(turn.pop("_uncertain_flags"))), 4
        )
        turn["speaker_assignment_uncertain"] = bool(
            turn["speaker"] == "Speaker uncertain"
            or turn["speaker_uncertainty_fraction"] >= 0.30
        )
    return turns

# def build_mono_speaker_turns(candidate, timeline):
#     turns = []
#     for word in candidate_words(candidate):
#         scores = speaker_overlap_scores(
#             word["start_seconds"], word["end_seconds"], timeline
#         )
#         primary_ratio = scores[0]["overlap_ratio"] if scores else 0.0
        
#         nearby_scores = speaker_overlap_scores(
#             max(0.0, float(word["start_seconds"]) - 0.20),
#             float(word["end_seconds"]) + 0.20,
#             timeline,
#         )
#         active_scores = [
#             score for score in scores
#             if score["overlap_seconds"] >= 0.04 and score["overlap_ratio"] >= 0.12
#         ]

#         if scores and primary_ratio >= 0.30:
#             primary = scores[0]["speaker"]
#         elif nearby_scores and nearby_scores[0]["overlap_ratio"] >= 0.20:
#             primary = nearby_scores[0]["speaker"]
#         else:
#             primary = "Speaker uncertain"
            
#         overlap = len(active_scores) > 1
#         close_second = bool(
#             len(active_scores) > 1
#             and active_scores[1]["overlap_ratio"]
#             >= 0.60 * active_scores[0]["overlap_ratio"]
#         )
#         assignment_uncertain = bool(
#             primary == "Speaker uncertain" or primary_ratio < 0.50 or close_second
#         )
#         active_speakers = [score["speaker"] for score in active_scores]
        # if not active_speakers and scores:
        #     active_speakers = [scores[0]["speaker"]]
        # if not active_speakers and primary != "Speaker uncertain":
        #     active_speakers = [primary]
        # if not active_speakers:
        #     active_speakers = ["Speaker uncertain"]

        # can_merge = bool(
        #     turns
        #     and turns[-1]["speaker"] == primary
        #     and turns[-1]["overlap_detected"] == overlap
        #     and turns[-1]["speaker_assignment_uncertain"] == assignment_uncertain
        #     and set(turns[-1]["active_speakers"]) == set(active_speakers)
        #     and word["start_seconds"] - turns[-1]["end_seconds"] <= 1.25
        #     and word["end_seconds"] - turns[-1]["start_seconds"] <= 20.0
        # )
        # if can_merge:
        #     turn = turns[-1]
        #     turn["end_seconds"] = word["end_seconds"]
        #     turn["transcript"] = normalize_spacing(turn["transcript"] + " " + word["text"])
        #     turn["active_speakers"] = sorted(
        #         set(turn["active_speakers"] + active_speakers)
        #     )
        #     turn["_probabilities"].append(float(word["probability"]))
        #     turn["_speaker_ratios"].append(float(primary_ratio))
        #     turn["_uncertain_flags"].append(float(assignment_uncertain))
        # else:
        #     turns.append({
        #         "start_seconds": word["start_seconds"],
        #         "end_seconds": word["end_seconds"],
        #         "source_track_index": 0,
        #         "speaker": primary,
        #         "active_speakers": active_speakers,
        #         "overlap_detected": overlap,
        #         "speaker_assignment_uncertain": assignment_uncertain,
        #         "transcript": word["text"],
        #         "whisper_language": candidate.get("detected_language"),
        #         "whisper_requested_language": candidate.get("requested_language"),
        #         "_probabilities": [float(word["probability"])],
        #         "_speaker_ratios": [float(primary_ratio)],
        #         "_uncertain_flags": [float(assignment_uncertain)],
        #     })
            
    # for turn in turns:
    #     turn["word_confidence"] = round(float(np.mean(turn.pop("_probabilities"))), 4)
    #     turn["speaker_overlap_ratio"] = round(
    #         float(np.mean(turn.pop("_speaker_ratios"))), 4
    #     )
    #     turn["speaker_uncertainty_fraction"] = round(
    #         float(np.mean(turn.pop("_uncertain_flags"))), 4
    #     )
    #     turn["speaker_assignment_uncertain"] = bool(
    #         turn["speaker"] == "Speaker uncertain"
    #         or turn["speaker_uncertainty_fraction"] >= 0.30
    #     )
    # return turns

def build_channel_turns(candidate, speaker, source_track_index):
    # Keep channel overlap decisions at word resolution. Long Whisper segments
    # often contain internal pauses and must not make the entire segment appear
    # simultaneous with the other call leg.
    return [
        {
            "start_seconds": word["start_seconds"],
            "end_seconds": word["end_seconds"],
            "source_track_index": source_track_index,
            "speaker": speaker,
            "active_speakers": [speaker],
            "overlap_detected": False,
            "speaker_assignment_uncertain": False,
            "speaker_overlap_ratio": 1.0,
            "speaker_uncertainty_fraction": 0.0,
            "transcript": word["text"],
            "word_confidence": round(float(word["probability"]), 4),
            "whisper_language": candidate.get("detected_language"),
            "whisper_requested_language": candidate.get("requested_language"),
        }
        for word in candidate_words(candidate)
        if normalize_spacing(word.get("text"))
    ]


def mark_channel_overlaps(records):
    ordered = sorted(records, key=lambda row: (row["start_seconds"], row["end_seconds"]))
    active_sets = [{item["speaker"]} for item in ordered]
    for index, item in enumerate(ordered):
        for other_index in range(index + 1, len(ordered)):
            other = ordered[other_index]
            if float(other["start_seconds"]) >= float(item["end_seconds"]):
                break
            if item["speaker"] == other["speaker"]:
                continue
            overlap = max(
                0.0,
                min(item["end_seconds"], other["end_seconds"])
                - max(item["start_seconds"], other["start_seconds"]),
            )
            if overlap > 0.04:
                active_sets[index].add(other["speaker"])
                active_sets[other_index].add(item["speaker"])

    for item, active in zip(ordered, active_sets):
        item["active_speakers"] = sorted(active)
        item["overlap_detected"] = len(active) > 1
        item["speaker_assignment_uncertain"] = False
        item["speaker_overlap_ratio"] = 1.0
        item["speaker_uncertainty_fraction"] = 0.0
    return ordered


configure_nemo_accuracy()
print(
    "NeMo is configured for batch-size-1 highest-context inference "
    "with calibrated speaker post-processing."
)


# ## Step 8 — Translate finalized turns only after Stage 1
# 
# IndicTrans2 translates normal English→Hindi and Hindi→English turns in small GPU batches after transcription and speaker labeling have already returned. A content-loss/repetition check flags failed Hindi/Hinglish output. Whole-track Whisper audio recovery is opt-in because it adds an expensive second ASR pass. There are no arbitrary text chunks and no voice synthesis.


def finalize_text(text, language_code):
    text = normalize_spacing(text)
    if language_code == "en" and text and text[-1] not in ".!?":
        text += "."
    elif language_code == "hi" and text and text[-1] not in ".!?।":
        text += "।"
    return normalize_spacing(text)


INCOMPLETE_END_TOKENS = {
    "to", "and", "or", "but", "because", "if", "that", "with", "for", "of",
    "the", "a", "an", "कि", "और", "या", "लेकिन", "क्योंकि", "अगर", "तो",
    "के", "की", "का",
}


def source_looks_incomplete(text):
    normalized = normalize_spacing(text)
    if not normalized or normalized[-1] in ".!?।…":
        return False
    tokens = re.findall(r"[A-Za-z0-9\u0900-\u097F']+", normalized.lower())
    return bool(tokens and tokens[-1] in INCOMPLETE_END_TOKENS)


def preserve_incomplete_marker(source, translated):
    translated = normalize_spacing(translated)
    if not translated or not source_looks_incomplete(source):
        return translated
    return translated.rstrip(" .!?।…") + "…"


def naturalize_short_confirmation(source, translated, direction):
    if direction != "en-hi":
        return translated
    canonical = re.sub(r"[^a-z0-9]+", " ", normalize_spacing(source).lower()).strip()
    positive = {
        "yes", "yeah", "yep", "yes i am", "yeah i am",
        "yes i do", "yeah i do", "yes i can", "yeah i can",
    }
    negative = {
        "no", "no i am not", "no im not", "no i dont",
        "no i do not", "no i cant", "no i cannot",
    }
    if canonical in positive:
        return "हाँ।"
    if canonical in negative:
        return "नहीं।"
    return translated


def detect_text_language(text, whisper_language=None):
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari and latin:
        return "hi-en", ("hi" if devanagari >= latin else "en"), "Hinglish"
    if devanagari:
        return "hi", "hi", "Hindi"
    if latin:
        if whisper_language == "hi":
            return "hi-Latn", "hi", "Hindi/Hinglish (Latin script)"
        return "en", "en", "English"
    if whisper_language == "hi":
        return "hi", "hi", "Hindi"
    return "en", "en", "English"


def translate_complete_batch(texts, direction):
    texts = [normalize_spacing(text) for text in texts]
    if not texts:
        return []
    if direction == "en-hi":
        source_texts = [finalize_text(text, "en") for text in texts]
        processor_args = {"src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}
        tokenizer, model, post_lang, target = EN_HI_TOKENIZER, EN_HI_MODEL, "hin_Deva", "hi"
    else:
        source_texts = [finalize_text(text, "hi") for text in texts]
        processor_args = {"src_lang": "hin_Deva", "tgt_lang": "eng_Latn"}
        tokenizer, model, post_lang, target = HI_EN_TOKENIZER, HI_EN_MODEL, "eng_Latn", "en"

    # Complete timestamped turns are batched as independent rows, never split.
    processed = INDIC_PROCESSOR.preprocess_batch(source_texts, **processor_args)
    encoded = tokenizer(
        processed,
        truncation=False,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    token_counts = encoded["attention_mask"].sum(dim=1).tolist()
    if max(token_counts, default=0) > 256:
        raise RuntimeError(
            "A timestamped turn exceeded IndicTrans2's 256-token input limit. "
            "Shorten the speaker-turn merge duration instead of silently truncating text."
        )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=384,
            num_beams=TRANSLATION_BEAM_SIZE,
            do_sample=False,
            # IndicTrans2 custom code predates Transformers' new cache API.
            # Disabling decoder KV caching keeps generation correct and avoids
            # a tuple-of-None past_key_values crash on current Colab versions.
            use_cache=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translated = INDIC_PROCESSOR.postprocess_batch(decoded, lang=post_lang)
    finalized = []
    for source, text in zip(texts, translated):
        value = finalize_text(collapse_consecutive_phrase_repeats(text), target)
        value = naturalize_short_confirmation(source, value, direction)
        value = preserve_incomplete_marker(source, value)
        finalized.append(value)
    return finalized


def translate_complete_turn(text, direction):
    if not normalize_spacing(text):
        return ""
    return translate_complete_batch([text], direction)[0]


def append_translation_warning(item, warning):
    warning = normalize_spacing(warning)
    if not warning:
        return
    existing = normalize_spacing(item.get("translation_quality_warning", ""))
    parts = [part.strip() for part in existing.split(";") if part.strip()]
    if warning not in parts:
        parts.append(warning)
    item["translation_quality_warning"] = "; ".join(parts)
    item["translation_review_required"] = True


def source_translation_review_reasons(item):
    source = normalize_spacing(item.get("transcript", ""))
    reasons = []
    if item.get("overlap_detected"):
        reasons.append("source contains overlapping speech")
    if item.get("speaker_assignment_uncertain"):
        reasons.append("speaker attribution is uncertain")
    if source_looks_incomplete(source):
        reasons.append("source turn appears incomplete")
    tokens = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", source.lower())
    if any(left == right for left, right in zip(tokens, tokens[1:])):
        reasons.append("source contains repeated adjacent words")
    if re.search(r"\b\d+[A-Za-z]+\b", source):
        reasons.append("source contains an ambiguous number/unit token")
    confidence = item.get("word_confidence")
    if confidence is not None:
        try:
            if float(confidence) < 0.50:
                reasons.append("source word confidence is low")
        except (TypeError, ValueError):
            pass
    return reasons


def translation_failure_reason(source, translated):
    source_words = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", normalize_spacing(source))
    output_words = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", normalize_spacing(translated))
    if not output_words:
        return "empty translation"
    if len(source_words) >= 10 and len(output_words) / max(1, len(source_words)) < 0.55:
        return "translation lost most source content"
    if len(output_words) >= 6 and repeated_ngram_ratio(translated, n=2) > 0.20:
        return "translation became repetitive"
    return None


def whisper_translate_turn(wav_path, start_seconds, end_seconds):
    # Decode only the finalized problematic turn. Whole-track translation
    # segments can span several speakers and cannot be mapped back reliably.
    clip_start = max(0.0, float(start_seconds) - 0.20)
    clip_end = max(clip_start + 0.10, float(end_seconds) + 0.20)
    result = ASR_MODEL.transcribe(
        str(wav_path),
        language="hi",
        task="translate",
        verbose=None,
        beam_size=WHISPER_TRANSLATION_BEAM_SIZE,
        patience=1.2,
        temperature=(0.0, 0.2),
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.60,
        word_timestamps=False,
        clip_timestamps=f"{clip_start:.3f},{clip_end:.3f}",
        fp16=True,
    )
    text = normalize_spacing(" ".join(
        segment.get("text", "")
        for segment in result.get("segments", [])
        if normalize_spacing(segment.get("text"))
    ))
    return finalize_text(collapse_consecutive_phrase_repeats(text), "en")


def translate_records_batch(records, tracks, enable_audio_fallback=False):
    enriched = [dict(item) for item in records]
    groups = {"en-hi": [], "hi-en": []}
    for index, item in enumerate(enriched):
        detected_code, route, detected_name = detect_text_language(item["transcript"], item.get("whisper_language"))
        direction = "hi-en" if route == "hi" else "en-hi"
        item["detected_language"] = detected_code
        item["detected_language_name"] = detected_name
        item["translation_direction"] = (
            "Hindi → English" if direction == "hi-en" else "English → Hindi"
        )
        item["translation_model"] = (
            "IndicTrans2 Hindi → English"
            if direction == "hi-en" else "IndicTrans2 English → Hindi"
        )
        groups[direction].append(index)

    for direction, indices in groups.items():
        for offset in range(0, len(indices), TEXT_TRANSLATION_BATCH_SIZE):
            batch_indices = indices[offset:offset + TEXT_TRANSLATION_BATCH_SIZE]
            translations = translate_complete_batch(
                [enriched[index]["transcript"] for index in batch_indices],
                direction,
            )
            for index, translated in zip(batch_indices, translations):
                enriched[index]["translation"] = translated

    # Preserve every transcript turn unchanged. Translation-only review flags
    # identify source conditions that can make a fluent translation misleading.
    for item in enriched:
        item["translation_review_required"] = False
        for reason in source_translation_review_reasons(item):
            append_translation_warning(item, reason)

    # IndicTrans2 is fast for normal written turns, but some noisy Hinglish
    # turns collapse into a short or repeated phrase. Only then spend a
    # targeted turn-level Whisper speech-translation pass when requested.
    unreliable_by_track = {}
    for index, item in enumerate(enriched):
        if item["translation_direction"] != "Hindi → English":
            continue
        reason = translation_failure_reason(item["transcript"], item.get("translation", ""))
        if item.get("detected_language") == "hi-Latn":
            reason = "Romanized Hindi needs original-audio translation for reliable English output"
        if reason:
            append_translation_warning(item, reason)
            unreliable_by_track.setdefault(int(item.get("source_track_index", 0)), []).append(index)

    fallback_passes = 0
    if not enable_audio_fallback:
        for indices in unreliable_by_track.values():
            for index in indices:
                enriched[index]["translation_quality_warning"] += "; slow audio fallback not requested"
        return enriched, fallback_passes

    for track_index, unreliable_indices in sorted(unreliable_by_track.items()):
        if track_index >= len(tracks):
            continue
        wav_path = tracks[track_index]["raw_path"]
        for index in unreliable_indices:
            item = enriched[index]
            recovered = whisper_translate_turn(
                wav_path,
                item["start_seconds"],
                item["end_seconds"],
            )
            fallback_passes += 1
            if recovered:
                item["translation"] = recovered
                item["translation_model"] = (
                    "Whisper large-v3 turn-level speech-translation fallback"
                )
                item["translation_quality_warning"] += (
                    "; replaced from the matching original-audio turn"
                )
    return enriched, fallback_passes


# ## Step 9


def comparable_text(text):
    return " ".join(
        token for token in (canonical_token(value) for value in normalize_spacing(text).split())
        if token
    )

def resolve_uncertain_speakers_globally(records):
    """Resolve earlier uncertainty only when later/nearby acoustic evidence agrees.

    NeMo already assigns stable speaker identities across the full recording. This
    second pass is allowed to revise an earlier uncertain fragment after later
    confirmed speech becomes available, but it never infers a human identity or
    resolves overlap. Conflicting neighbours remain explicitly uncertain.
    """
    resolved = 0
    confirmed = [
        (index, item) for index, item in enumerate(records)
        if item.get("speaker") not in (None, "Speaker uncertain")
        and not item.get("speaker_assignment_uncertain")
        and not item.get("overlap_detected")
    ]
    for index, item in enumerate(records):
        if not (
            item.get("speaker_assignment_uncertain")
            or item.get("speaker") == "Speaker uncertain"
        ):
            continue
        if item.get("overlap_detected"):
            continue

        track_index = item.get("source_track_index")
        previous = next((
            row for row_index, row in reversed(confirmed)
            if row_index < index and row.get("source_track_index") == track_index
        ), None)
        following = next((
            row for row_index, row in confirmed
            if row_index > index and row.get("source_track_index") == track_index
        ), None)
        left_gap = (
            item["start_seconds"] - previous["end_seconds"]
            if previous is not None else float("inf")
        )
        right_gap = (
            following["start_seconds"] - item["end_seconds"]
            if following is not None else float("inf")
        )
        nearby_previous = previous if -0.10 <= left_gap <= 1.50 else None
        nearby_following = following if -0.10 <= right_gap <= 1.50 else None

        acoustic_candidates = {
            speaker for speaker in item.get("active_speakers", [])
            if speaker not in (None, "Speaker uncertain")
        }
        if item.get("speaker") not in (None, "Speaker uncertain"):
            acoustic_candidates.add(item["speaker"])

        chosen = None
        evidence = None
        if (
            nearby_previous is not None
            and nearby_following is not None
            and nearby_previous["speaker"] == nearby_following["speaker"]
            and nearby_previous["speaker"] in acoustic_candidates
        ):
            chosen = nearby_previous["speaker"]
            evidence = "matching confirmed speech before and after"
        else:
            one_sided = nearby_previous or nearby_following
            opposite = nearby_following if nearby_previous is not None else nearby_previous
            single_acoustic = next(iter(acoustic_candidates)) if len(acoustic_candidates) == 1 else None
            if (
                one_sided is not None
                and opposite is None
                and single_acoustic == one_sided["speaker"]
                and float(item.get("speaker_overlap_ratio", 0.0)) >= 0.20
                and item["end_seconds"] - item["start_seconds"] <= 4.0
            ):
                chosen = single_acoustic
                evidence = (
                    "confirmed later speech plus acoustic candidate"
                    if nearby_following is not None
                    else "confirmed earlier speech plus acoustic candidate"
                )

        if chosen is None:
            continue
        item["speaker"] = chosen
        item["speaker_assignment_uncertain"] = False
        item["speaker_assignment_recovered"] = True
        item["speaker_resolution_method"] = evidence
        item["active_speakers"] = sorted(acoustic_candidates | {chosen})
        resolved += 1
    return records, resolved




# Short Duration: The spoken audio lasts less than max_micro_duration (default 0.8 seconds).
# Word Count: Contains 2 or fewer words (e.g., "So,", "Can?", "I").
# Short Time Gap: The gap between the micro-utterance and the next/previous turn is less than max_gap (default 1.5 seconds).
# Trigger Flags: The turn has an [uncertain] speaker tag OR ends with a dangling grammar word (like so, and, can, if, i, the, a) OR ends with a comma ,.
# 1. Algorithmic Post-Processing (Highly Recommended & Fastest)
def smooth_micro_utterances(turns, max_gap=1.5, max_micro_duration=0.8):
    """
    Merges orphaned micro-utterances (e.g., "So,", "Can?", "I") into adjacent 
    confident speaker turns based on time gaps and syntactic cues.
    """
    if not turns:
        return []

    forward_dangling = {"so", "and", "but", "if", "can", "i", "the", "a", "is", "are", "am", "my"}
    smoothed = []
    skip_next = False

    for i in range(len(turns)):
        if skip_next:
            skip_next = False
            continue

        current = dict(turns[i])
        current_text = current.get("transcript", "").strip()
        current_words = [w.strip(".,?!\"'") for w in current_text.lower().split()]
        
        # RULE 1: FORWARD MERGE
        if i + 1 < len(turns):
            next_turn = dict(turns[i+1])
            gap_forward = next_turn["start_seconds"] - current["end_seconds"]
            current_duration = current["end_seconds"] - current["start_seconds"]
            
            is_micro_forward = (
                current_duration < max_micro_duration 
                and len(current_words) <= 2
                and gap_forward < max_gap
                and (
                    "uncertain" in current.get("speaker", "").lower() 
                    or (current_words and current_words[-1] in forward_dangling)
                    or current_text.endswith(",")
                )
            )

            if is_micro_forward:
                next_turn["start_seconds"] = current["start_seconds"]
                if current_text.endswith(","):
                    next_turn["transcript"] = f"{current_text} {next_turn.get('transcript', '').strip()}"
                else:
                    import re
                    clean_current = re.sub(r'[^\w\s]$', '', current_text)
                    next_turn["transcript"] = f"{clean_current} {next_turn.get('transcript', '').strip()}"
                
                if "uncertain" in next_turn.get("speaker", "").lower() and "uncertain" not in current.get("speaker", "").lower():
                    next_turn["speaker"] = current["speaker"]
                    
                smoothed.append(next_turn)
                skip_next = True
                continue

        # RULE 2: BACKWARD MERGE
        if smoothed:
            prev_turn = smoothed[-1]
            gap_backward = current["start_seconds"] - prev_turn["end_seconds"]
            current_duration = current["end_seconds"] - current["start_seconds"]
            
            is_micro_backward = (
                current_duration < max_micro_duration
                and len(current_words) <= 2
                and gap_backward < max_gap
                and "uncertain" in current.get("speaker", "").lower()
            )

            if is_micro_backward:
                prev_turn["end_seconds"] = current["end_seconds"]
                prev_turn["transcript"] = f"{prev_turn.get('transcript', '').strip()} {current_text}"
                continue 
                
        smoothed.append(current)

    return smoothed

# CONTEXT_CORRECTION_PROFILES = {
#     "personal-loan": {
#         "minimum_evidence_score": 5,
#         "evidence_weights": {
#             "personal loan": 4,
#             "loan amount": 3,
#             "loan offer": 3,
#             "eligibility": 2,
#             "salaried": 1,
#             "self-employed": 1,
#             "self employed": 1,
#             "emi": 2,
#             "kyc": 2,
#         },
#         "canonical_terms": {
#             "loan": {
#                 "minimum_similarity": 0.74,
#                 "acoustic_aliases": {
#                     "look": 0.86,
#                     "lone": 0.94,
#                     "loam": 0.82,
#                 },
#                 "local_cues": {
#                     "purpose", "amount", "offer", "eligibility",
#                     "application", "apply", "personal",
#                 },
#             },
#         },
#         "amount_cues": {
#             "amount", "how much", "loan", "borrow", "looking for", "required",
#         },
#         "regional_amount_unit": {
#             "ambiguous_suffixes": {"x", "×"},
#             "singular": "lakh",
#             "plural": "lakhs",
#         },
#     },
# }


def _bounded_edit_distance(left, right):
    left, right = str(left).lower(), str(right).lower()
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _token_similarity(left, right):
    longest = max(len(str(left)), len(str(right)), 1)
    return 1.0 - (_bounded_edit_distance(left, right) / longest)


# def _infer_correction_profile(conversation):
#     normalized = str(conversation or "").lower()
#     scored = []
#     for name, profile in CONTEXT_CORRECTION_PROFILES.items():
#         score = sum(
#             weight for phrase, weight in profile["evidence_weights"].items()
#             if phrase in normalized
#         )
#         scored.append((score, name, profile))
#     score, name, profile = max(scored, default=(0, "general", None))
#     if profile is None or score < profile["minimum_evidence_score"]:
#         return "general", score, None
#     return name, score, profile
def _infer_correction_profile(conversation):
    _, domain_info = classify_and_detect_anomalies_flashtext(conversation)
    if domain_info:
        cat_id = domain_info["category"]
        return cat_id, domain_info["score"], MASTER_CALL_REGISTRY[cat_id]
    return "general", 0, None

# def _replace_contextual_near_terms(text, profile):
#     """Repair a near-homophone only when local syntax and domain evidence agree."""
#     if not profile:
#         return text, []
#     source = str(text or "")
#     lowered = source.lower()
#     replacements = []
#     token_pattern = re.compile(r"\b[A-Za-z][A-Za-z'-]{1,20}\b")
#     matches = list(token_pattern.finditer(source))

#     for match in reversed(matches):
#         observed = match.group(0)
#         observed_lower = observed.lower()
#         for canonical, term_config in profile["canonical_terms"].items():
#             if observed_lower == canonical:
#                 continue
#             orthographic_similarity = _token_similarity(observed_lower, canonical)
#             acoustic_similarity = term_config.get("acoustic_aliases", {}).get(
#                 observed_lower, 0.0
#             )
#             similarity = max(orthographic_similarity, acoustic_similarity)
#             if similarity < term_config["minimum_similarity"]:
#                 continue

#             prefix = lowered[max(0, match.start() - 36):match.start()]
#             suffix = lowered[match.end():min(len(lowered), match.end() + 36)]
#             local_window = prefix + " " + suffix
#             cue_supported = any(
#                 cue in local_window for cue in term_config["local_cues"]
#             )
#             noun_position = bool(
#                 re.search(r"\b(?:your|the|a|an|this|that|personal)\s+$", prefix)
#                 or re.match(r"^\s+(?:amount|offer|application)\b", suffix)
#             )
#             if not (cue_supported and noun_position):
#                 continue

#             replacement = canonical.capitalize() if observed[:1].isupper() else canonical
#             source = source[:match.start()] + replacement + source[match.end():]
#             replacements.append({
#                 "observed": observed,
#                 "canonical": replacement,
#                 "similarity": round(similarity, 3),
#                 "evidence_source": (
#                     "configured-acoustic-alias"
#                     if acoustic_similarity > orthographic_similarity
#                     else "orthographic-neighbor"
#                 ),
#             })
#             lowered = source.lower()
#             break

#     return source, list(reversed(replacements))




# def apply_context_aware_transcript_corrections(records):
#     """Conservative domain-aware normalization with raw ASR and an audit trail."""
#     if not records:
#         return records, {
#             "context_corrected_turns": 0,
#             "context_correction_count": 0,
#             "context_correction_types": {},
#             "raw_transcript_preserved": True,
#         }

    # raw_texts = [normalize_spacing(item.get("transcript", "")) for item in records]
    # conversation = " ".join(raw_texts)
    # profile_name, profile_score, profile = _infer_correction_profile(conversation)
    # corrected_records = []
    # correction_type_counts = {}
    # corrected_turns = 0
    # total_corrections = 0

    # for index, source_item in enumerate(records):
    #     item = dict(source_item)
    #     raw_text = raw_texts[index]
    #     corrected = raw_text
    #     corrections = []
    #     nearby_text = " ".join(
    #         raw_texts[max(0, index - 1):min(len(raw_texts), index + 2)]
    #     ).lower()

        # def record_correction(rule_name, before, after, reason, evidence=None):
        #     nonlocal total_corrections
        #     if before == after:
        #         return
        #     detail = {
        #         "rule": rule_name,
        #         "from": before,
        #         "to": after,
        #         "reason": reason,
        #     }
        #     if evidence:
        #         detail["evidence"] = evidence
        #     corrections.append(detail)
        #     correction_type_counts[rule_name] = (
        #         correction_type_counts.get(rule_name, 0) + 1
        #     )
        #     total_corrections += 1

        # before = corrected
        # corrected = re.sub(
        #     r"(?i)\b([A-Za-z']+)(?:\s+\1\b)+",
        #     r"\1",
        #     corrected,
        # )
        # record_correction(
        #     "adjacent-word-deduplication",
        #     before,
        #     corrected,
        #     "the same lexical token was repeated consecutively",
        # )

        # before = corrected
        # corrected = re.sub(
        #     r"(?i)\bself\s*-\s*employed\b",
        #     "self-employed",
        #     corrected,
        # )
        # record_correction(
        #     "compound-word-normalization",
        #     before,
        #     corrected,
        #     "normalized spacing inside a standard compound word",
        # )

        # before = corrected
        # corrected, term_evidence = _replace_contextual_near_terms(corrected, profile)
        # record_correction(
        #     "contextual-near-term",
        #     before,
        #     corrected,
        #     "domain evidence, local cue words, noun position, and edit similarity agreed",
        #     term_evidence,
        # )

        # if profile:
        #     unit_config = profile["regional_amount_unit"]
        #     amount_context = any(
        #         cue in nearby_text for cue in profile["amount_cues"]
        #     )

        #     if amount_context:
        #         suffixes = "|".join(
        #             re.escape(value)
        #             for value in sorted(unit_config["ambiguous_suffixes"])
        #         )

        #         def replace_regional_amount(match):
        #             number_text = match.group(1)
        #             unit = (
        #                 unit_config["singular"]
        #                 if number_text in {"1", "1.0"}
        #                 else unit_config["plural"]
        #             )
        #             return f"{number_text} {unit}"

        #         before = corrected
        #         corrected = re.sub(
        #             rf"(?i)\b(\d+(?:\.\d+)?)\s*(?:{suffixes})\b",
        #             replace_regional_amount,
        #             corrected,
            #     )
            #     record_correction(
            #         "regional-amount-unit",
            #         before,
            #         corrected,
            #         "a numeric suffix was ambiguous, while neighboring turns established an Indian loan-amount slot",
            #     )

            # salaried = re.search(r"(?i)\bsalaried\b", corrected)
            # self_employed = re.search(
            #     r"(?i)\bself-employed\b",
            #     corrected,
            # )
            # question_like = bool(re.search(
            #     r"(?i)^\s*(?:okay[\s,]+)?(?:uh[\s,]+)?"
            #     r"(?:are|is|do|does|can|could|may|how|what)\b",
            #     corrected,
            # ))
            # if (
            #     salaried and self_employed and question_like
            #     and salaried.end() < self_employed.start()
            # ):
            #     noisy_span = corrected[salaried.end():self_employed.start()]
            #     noise_tokens = set(re.findall(r"[A-Za-z']+", noisy_span.lower()))
            #     allowed_noise = {
            #         "or", "if", "i", "i'm", "im", "in", "salaried", "self",
            #     }
            #     if "or" in noise_tokens and noise_tokens <= allowed_noise:
            #         before = corrected
            #         corrected = (
            #             corrected[:salaried.end()]
            #             + " or self-employed"
            #             + corrected[self_employed.end():]
            #         )
            #         record_correction(
            #             "employment-option-schema",
            #             before,
            #             corrected,
            #             "two recognized employment-status values formed one question; only ASR filler inside that option span was removed",
            #         )

        # before = corrected
        # corrected = re.sub(
        #     r"(?i)\bi(?=\b|['’](?:m|d|ll|ve|re)\b)",
        #     "I",
        #     corrected,
        # )
        # record_correction(
        #     "first-person-pronoun-casing",
        #     before,
        #     corrected,
        #     "normalized the English first-person pronoun without changing lexical content",
        # )

        # before = corrected
        # corrected = re.sub(
        #     r"(?i)^\s*(okay|well|so)\s+(uh|um)\s+",
        #     lambda match: (
        #         f"{match.group(1).capitalize()}, "
        #         f"{match.group(2).lower()}, "
        #     ),
        #     corrected,
        # )
        # record_correction(
        #     "discourse-marker-punctuation",
        #     before,
        #     corrected,
        #     "punctuated preserved discourse and hesitation markers",
        # )

    #     corrected = normalize_spacing(corrected)
    #     question_like = bool(re.search(
    #         r"(?i)^\s*(?:okay[\s,]+)?(?:uh[\s,]+)?"
    #         r"(?:are|is|do|does|did|can|could|may|would|will|"
    #         r"how|what|why|when|where|who)\b",
    #         corrected,
    #     ))
    #     if corrected:
    #         before = corrected
    #         corrected = corrected[:1].upper() + corrected[1:]
    #         if question_like:
    #             corrected = re.sub(r"[.!]*$", "?", corrected)
    #         record_correction(
    #             "sentence-boundary-formatting",
    #             before,
    #             corrected,
    #             "capitalization and terminal punctuation followed the detected utterance type",
    #         )

    #     item["raw_transcript"] = raw_text
    #     item["transcript"] = corrected
    #     item["contextual_corrections"] = corrections
    #     item["context_correction_applied"] = bool(corrections)
    #     item["postprocessing_profile"] = (
    #         f"{profile_name}-evidence-gated-v2"
    #         if profile else "general-structure-v2"
    #     )
    #     if corrections:
    #         corrected_turns += 1
    #     corrected_records.append(item)

    # return corrected_records, {
    #     "context_domain": profile_name,
    #     "context_domain_score": profile_score,
    #     "context_corrected_turns": corrected_turns,
    #     "context_correction_count": total_corrections,
    #     "context_correction_types": correction_type_counts,
    #     "raw_transcript_preserved": True,
    # }

def _replace_contextual_near_terms(text, profile: Optional[MasterDomainConfig]):
    """Repair a near-homophone only when local syntax and domain evidence agree."""
    if not profile or not profile.canonical_terms:
        return text, []
    source = str(text or "")
    lowered = source.lower()
    replacements = []
    token_pattern = re.compile(r"\b[A-Za-z][A-Za-z'-]{1,20}\b")
    matches = list(token_pattern.finditer(source))

    for match in reversed(matches):
        observed = match.group(0)
        observed_lower = observed.lower()
        for canonical_key, term_config in profile.canonical_terms.items():
            if observed_lower == term_config.canonical:
                continue
            orthographic_similarity = _token_similarity(observed_lower, term_config.canonical)
            acoustic_similarity = term_config.acoustic_aliases.get(observed_lower, 0.0)
            similarity = max(orthographic_similarity, acoustic_similarity)
            
            if similarity < term_config.minimum_similarity:
                continue

            prefix = lowered[max(0, match.start() - 36):match.start()]
            suffix = lowered[match.end():min(len(lowered), match.end() + 36)]
            local_window = prefix + " " + suffix
            cue_supported = any(cue in local_window for cue in term_config.local_cues)
            noun_position = bool(
                re.search(r"\b(?:your|the|a|an|this|that|personal)\s+$", prefix)
                or re.match(r"^\s+(?:amount|offer|application)\b", suffix)
            )
            if not (cue_supported and noun_position):
                continue

            replacement = term_config.canonical.capitalize() if observed[:1].isupper() else term_config.canonical
            source = source[:match.start()] + replacement + source[match.end():]
            replacements.append({
                "observed": observed,
                "canonical": replacement,
                "similarity": round(similarity, 3),
                "evidence_source": (
                    "configured-acoustic-alias"
                    if acoustic_similarity > orthographic_similarity
                    else "orthographic-neighbor"
                ),
            })
            lowered = source.lower()
            break

    return source, list(reversed(replacements))


def apply_context_aware_transcript_corrections(records):
    """Conservative domain-aware normalization with raw ASR and an audit trail."""
    if not records:
        return records, {
            "context_corrected_turns": 0,
            "context_correction_count": 0,
            "context_correction_types": {},
            "raw_transcript_preserved": True,
        }

    raw_texts = [normalize_spacing(item.get("transcript", "")) for item in records]
    conversation = " ".join(raw_texts)
    profile_name, profile_score, profile = _infer_correction_profile(conversation)
    corrected_records = []
    correction_type_counts = {}
    corrected_turns = 0
    total_corrections = 0

    for index, source_item in enumerate(records):
        item = dict(source_item)
        raw_text = raw_texts[index]
        corrected = raw_text
        corrections = []
        nearby_text = " ".join(
            raw_texts[max(0, index - 1):min(len(raw_texts), index + 2)]
        ).lower()

        def record_correction(rule_name, before, after, reason, evidence=None):
            nonlocal total_corrections
            if before == after:
                return
            detail = {
                "rule": rule_name,
                "from": before,
                "to": after,
                "reason": reason,
            }
            if evidence:
                detail["evidence"] = evidence
            corrections.append(detail)
            correction_type_counts[rule_name] = (
                correction_type_counts.get(rule_name, 0) + 1
            )
            total_corrections += 1

        # Rule 1: Consecutive Word Deduplication
        before = corrected
        corrected = re.sub(
            r"(?i)\b([A-Za-z']+)(?:\s+\1\b)+",
            r"\1",
            corrected,
        )
        record_correction(
            "adjacent-word-deduplication",
            before,
            corrected,
            "the same lexical token was repeated consecutively",
        )

        # Rule 2: Standard Compound Word Spacing
        before = corrected
        corrected = re.sub(
            r"(?i)\bself\s*-\s*employed\b",
            "self-employed",
            corrected,
        )
        record_correction(
            "compound-word-normalization",
            before,
            corrected,
            "normalized spacing inside a standard compound word",
        )

        # Rule 3: Domain-Aware Acoustic Near-Term Replacement
        before = corrected
        corrected, term_evidence = _replace_contextual_near_terms(corrected, profile)
        record_correction(
            "contextual-near-term",
            before,
            corrected,
            "domain evidence, local cue words, noun position, and edit similarity agreed",
            term_evidence,
        )

        # Rule 4: Regional Amount Unit Normalization (Dataclass Refactored)
        if profile and profile.amount_config:
            unit_config = profile.amount_config
            amount_context = any(
                cue in nearby_text for cue in unit_config.amount_cues
            )

            if amount_context:
                suffixes = "|".join(
                    re.escape(value)
                    for value in sorted(unit_config.ambiguous_suffixes)
                )

                def replace_regional_amount(match):
                    number_text = match.group(1)
                    unit = (
                        unit_config.singular
                        if number_text in {"1", "1.0"}
                        else unit_config.plural
                    )
                    return f"{number_text} {unit}"

                before = corrected
                corrected = re.sub(
                    rf"(?i)\b(\d+(?:\.\d+)?)\s*(?:{suffixes})\b",
                    replace_regional_amount,
                    corrected,
                )
                record_correction(
                    "regional-amount-unit",
                    before,
                    corrected,
                    "a numeric suffix was ambiguous, while neighboring turns established an Indian loan-amount slot",
                )

            salaried = re.search(r"(?i)\bsalaried\b", corrected)
            self_employed = re.search(
                r"(?i)\bself-employed\b",
                corrected,
            )
            question_like = bool(re.search(
                r"(?i)^\s*(?:okay[\s,]+)?(?:uh[\s,]+)?"
                r"(?:are|is|do|does|can|could|may|how|what)\b",
                corrected,
            ))
            if (
                salaried and self_employed and question_like
                and salaried.end() < self_employed.start()
            ):
                noisy_span = corrected[salaried.end():self_employed.start()]
                noise_tokens = set(re.findall(r"[A-Za-z']+", noisy_span.lower()))
                allowed_noise = {
                    "or", "if", "i", "i'm", "im", "in", "salaried", "self",
                }
                if "or" in noise_tokens and noise_tokens <= allowed_noise:
                    before = corrected
                    corrected = (
                        corrected[:salaried.end()]
                        + " or self-employed"
                        + corrected[self_employed.end():]
                    )
                    record_correction(
                        "employment-option-schema",
                        before,
                        corrected,
                        "two recognized employment-status values formed one question; only ASR filler inside that option span was removed",
                    )

        # Rule 5: Pronoun and Discourse Formatting
        before = corrected
        corrected = re.sub(
            r"(?i)\bi(?=\b|['’](?:m|d|ll|ve|re)\b)",
            "I",
            corrected,
        )
        record_correction(
            "first-person-pronoun-casing",
            before,
            corrected,
            "normalized the English first-person pronoun without changing lexical content",
        )

        before = corrected
        corrected = re.sub(
            r"(?i)^\s*(okay|well|so)\s+(uh|um)\s+",
            lambda match: (
                f"{match.group(1).capitalize()}, "
                f"{match.group(2).lower()}, "
            ),
            corrected,
        )
        record_correction(
            "discourse-marker-punctuation",
            before,
            corrected,
            "punctuated preserved discourse and hesitation markers",
        )

        corrected = normalize_spacing(corrected)
        question_like = bool(re.search(
            r"(?i)^\s*(?:okay[\s,]+)?(?:uh[\s,]+)?"
            r"(?:are|is|do|does|did|can|could|may|would|will|"
            r"how|what|why|when|where|who)\b",
            corrected,
        ))
        if corrected:
            before = corrected
            corrected = corrected[:1].upper() + corrected[1:]
            if question_like:
                corrected = re.sub(r"[.!]*$", "?", corrected)
            record_correction(
                "sentence-boundary-formatting",
                before,
                corrected,
                "capitalization and terminal punctuation followed the detected utterance type",
            )

        item["raw_transcript"] = raw_text
        item["transcript"] = corrected
        item["contextual_corrections"] = corrections
        item["context_correction_applied"] = bool(corrections)
        item["postprocessing_profile"] = (
            f"{profile_name}-evidence-gated-v2"
            if profile else "general-structure-v2"
        )
        if corrections:
            corrected_turns += 1
        corrected_records.append(item)

    return corrected_records, {
        "context_domain": profile_name,
        "context_domain_score": profile_score,
        "context_corrected_turns": corrected_turns,
        "context_correction_count": total_corrections,
        "context_correction_types": correction_type_counts,
        "raw_transcript_preserved": True,
    }
    
def postprocess_records(records, media_duration):
    """Conservative timestamp, duplicate-loop and turn cleanup."""
    stats = {
        "invalid_records_removed": 0,
        "overlapping_duplicates_removed": 0,
        "verbatim_repetitions_preserved": True,
        "global_uncertain_speakers_resolved": 0,
        "bridged_uncertain_fragments_reassigned": 0,
        "adjacent_turns_merged": 0,
    }
    normalized = []
    for source_item in sorted(records, key=lambda row: (row["start_seconds"], row["end_seconds"])):
        item = dict(source_item)
        item["start_seconds"] = round(max(0.0, min(float(media_duration), float(item["start_seconds"]))), 3)
        item["end_seconds"] = round(max(0.0, min(float(media_duration), float(item["end_seconds"]))), 3)
        item["transcript"] = normalize_spacing(item.get("transcript", ""))
        if not item["transcript"] or item["end_seconds"] - item["start_seconds"] < 0.04:
            stats["invalid_records_removed"] += 1
            continue
        normalized.append(item)

    # 1. This is your existing global resolution line
    normalized, globally_resolved = resolve_uncertain_speakers_globally(normalized)
    stats["global_uncertain_speakers_resolved"] = globally_resolved

    # 2. ADD THIS NEW LINE RIGHT HERE 
    # normalized = smooth_micro_utterances(normalized)

    for index, item in enumerate(normalized):
        is_uncertain = bool(
            item.get("speaker_assignment_uncertain")
            or item.get("speaker") == "Speaker uncertain"
        )
        if not is_uncertain or index == 0 or index + 1 >= len(normalized):
            continue
        previous = normalized[index - 1]
        following = normalized[index + 1]
        duration = item["end_seconds"] - item["start_seconds"]
        word_count = len(item["transcript"].split())
        left_gap = item["start_seconds"] - previous["end_seconds"]
        right_gap = following["start_seconds"] - item["end_seconds"]
        same_confirmed_speaker = bool(
            previous.get("speaker") == following.get("speaker")
            and previous.get("speaker") not in (None, "Speaker uncertain")
            and not previous.get("speaker_assignment_uncertain")
            and not following.get("speaker_assignment_uncertain")
            and previous.get("source_track_index") == item.get("source_track_index")
            == following.get("source_track_index")
        )
        acoustic_candidates = {
            speaker for speaker in item.get("active_speakers", [])
            if speaker not in (None, "Speaker uncertain")
        }
        no_conflicting_acoustic_speaker = bool(
            not acoustic_candidates
            or acoustic_candidates == {previous.get("speaker")}
        )
        if (
            same_confirmed_speaker
            and no_conflicting_acoustic_speaker
            and not item.get("overlap_detected")
            and not previous.get("overlap_detected")
            and not following.get("overlap_detected")
            and duration <= 3.0
            and word_count <= 8
            and -0.10 <= left_gap <= 0.75
            and -0.10 <= right_gap <= 0.75
        ):
            item["speaker"] = previous["speaker"]
            item["speaker_assignment_uncertain"] = False
            item["speaker_assignment_recovered"] = True
            item["speaker_resolution_method"] = (
                "same confirmed speaker on both sides; no conflicting acoustic evidence"
            )
            item["active_speakers"] = [previous["speaker"]]
            stats["bridged_uncertain_fragments_reassigned"] += 1

    cleaned = []
    for item in normalized:
        if cleaned and cleaned[-1].get("speaker") == item.get("speaker"):
            previous = cleaned[-1]
            overlap = max(
                0.0,
                min(previous["end_seconds"], item["end_seconds"])
                - max(previous["start_seconds"], item["start_seconds"]),
            )
            shorter = max(
                0.04,
                min(
                    previous["end_seconds"] - previous["start_seconds"],
                    item["end_seconds"] - item["start_seconds"],
                ),
            )
            similarity = SequenceMatcher(
                None, comparable_text(previous["transcript"]), comparable_text(item["transcript"])
            ).ratio()
            if overlap / shorter >= 0.35 and similarity >= 0.92:
                previous["end_seconds"] = max(previous["end_seconds"], item["end_seconds"])
                previous["active_speakers"] = sorted(set(
                    previous.get("active_speakers", []) + item.get("active_speakers", [])
                ))
                previous["word_confidence"] = round(max(
                    float(previous.get("word_confidence", 0.0)),
                    float(item.get("word_confidence", 0.0)),
                ), 4)
                stats["overlapping_duplicates_removed"] += 1
                continue

            gap = item["start_seconds"] - previous["end_seconds"]
            combined_duration = item["end_seconds"] - previous["start_seconds"]
            can_merge = bool(
                -0.10 <= gap <= 1.20
                and combined_duration <= 20.0
                and previous.get("source_track_index") == item.get("source_track_index")
                and previous.get("overlap_detected") == item.get("overlap_detected")
                and set(previous.get("active_speakers", [])) == set(item.get("active_speakers", []))
                and previous.get("speaker_assignment_uncertain")
                == item.get("speaker_assignment_uncertain")
            )
            if can_merge:
                previous["end_seconds"] = max(previous["end_seconds"], item["end_seconds"])
                previous["transcript"] = normalize_spacing(
                    previous["transcript"] + " " + item["transcript"]
                )
                previous["overlap_detected"] = bool(
                    previous.get("overlap_detected") or item.get("overlap_detected")
                )
                previous["speaker_assignment_recovered"] = bool(
                    previous.get("speaker_assignment_recovered")
                    or item.get("speaker_assignment_recovered")
                )
                previous["active_speakers"] = sorted(set(
                    previous.get("active_speakers", []) + item.get("active_speakers", [])
                ))
                for field in ("word_confidence", "speaker_overlap_ratio", "speaker_uncertainty_fraction"):
                    if field in previous or field in item:
                        previous[field] = round(float(np.mean([
                            float(previous.get(field, 0.0)), float(item.get(field, 0.0))
                        ])), 4)
                stats["adjacent_turns_merged"] += 1
                continue
        cleaned.append(item)

    cleaned, context_stats = apply_context_aware_transcript_corrections(cleaned)
    stats.update(context_stats)
    return cleaned, stats


def release_transient_gpu_memory():
    """Release request-scoped CUDA allocations without unloading the models."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def format_clock(seconds):
    milliseconds = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_results(records):
    transcript_lines, translation_lines = [], []
    for item in records:
        stamp = f"[{format_clock(item['start_seconds'])}–{format_clock(item['end_seconds'])}]"
        overlap = " [overlap detected]" if item.get("overlap_detected") else ""
        uncertain = " [speaker uncertain]" if item.get("speaker_assignment_uncertain") else ""
        transcript_lines.append(
            f"{stamp} {item['speaker']}{overlap}{uncertain}: {item['transcript']}"
        )
        translation_lines.append(
            f"{stamp} {item['speaker']} · {item.get('translation_direction', 'Translation off')}: "
            f"{item.get('translation', '')}"
        )
    return "\n\n".join(transcript_lines), "\n\n".join(translation_lines)


# def join_conversation_fragments(left, right):
#     """Join already-recognized fragments without inventing words or punctuation."""
#     left = normalize_spacing(left or "")
#     right = normalize_spacing(right or "")
#     if not left:
#         return right
#     if not right:
#         return left
#     if right[0] in ",.;:!?)]}%।":
#         return left.rstrip() + right
#     return left.rstrip() + " " + right.lstrip()


# def format_conversation(records):
#     """Readable speaker paragraphs; structured records and timestamps stay unchanged."""
#     groups = []
#     for item in records:
#         label = item["speaker"]
#         if item.get("overlap_detected"):
#             label += " [overlap]"
#         if item.get("speaker_assignment_uncertain"):
#             label += " [uncertain]"

#         start = float(item["start_seconds"])
#         end = float(item["end_seconds"])
#         can_continue = bool(
#             groups
#             and groups[-1]["label"] == label
#             and groups[-1]["source_track_index"] == item.get("source_track_index")
#             and -0.10 <= start - groups[-1]["end_seconds"] <= 2.0
#         )
#         if can_continue:
#             group = groups[-1]
#             group["end_seconds"] = max(group["end_seconds"], end)
#             group["transcript"] = join_conversation_fragments(
#                 group["transcript"], item.get("transcript", "")
#             )
#             group["translation"] = join_conversation_fragments(
#                 group["translation"], item.get("translation", "")
#             )
#         else:
#             groups.append({
#                 "label": label,
#                 "source_track_index": item.get("source_track_index"),
#                 "end_seconds": end,
#                 "transcript": normalize_spacing(item.get("transcript", "")),
#                 "translation": normalize_spacing(item.get("translation", "")),
#             })

#     transcript_lines = [
#         f"{group['label']}: {group['transcript']}" for group in groups
#     ]
#     translation_lines = [
#         f"{group['label']}: {group['translation']}" for group in groups
#     ]
#     return "\n\n".join(transcript_lines), "\n\n".join(translation_lines)
def join_conversation_fragments(left, right):
    """Join already-recognized fragments without inventing words or punctuation."""
    left = normalize_spacing(left or "")
    right = normalize_spacing(right or "")
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?)]}%।":
        return left.rstrip() + right
    return left.rstrip() + " " + right.lstrip()

def format_conversation(records):
    """
    Readable speaker paragraphs. 
    Clubs continuous speech together and breaks naturally on interruptions.
    """
    groups = []
    
    for item in records:
        # We drop the [overlap] tag from the UI label for a cleaner chat view, 
        # but keep [uncertain] if the model wasn't sure who spoke.
        label = item["speaker"]
        if item.get("speaker_assignment_uncertain"):
            label += " [uncertain]"

        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        transcript = normalize_spacing(item.get("transcript", ""))
        translation = normalize_spacing(item.get("translation", ""))
        
        if not transcript:
            continue

        # CLUBBING LOGIC: 
        # Group this text with the previous text IF:
        # 1. It is the exact same speaker.
        # 2. No other speaker interrupted in between.
        # 3. The time gap between their sentences is less than 3.5 seconds.
        can_continue = False
        if groups:
            last_group = groups[-1]
            is_same_speaker = last_group["label"] == label
            is_same_track = last_group.get("source_track_index") == item.get("source_track_index")
            time_gap = start - last_group["end_seconds"]
            
            if is_same_speaker and is_same_track and time_gap <= 3.5:
                can_continue = True

        if can_continue:
            # Club the text together
            group = groups[-1]
            group["end_seconds"] = max(group["end_seconds"], end)
            group["transcript"] = join_conversation_fragments(group["transcript"], transcript)
            group["translation"] = join_conversation_fragments(group["translation"], translation)
        else:
            # The speaker changed (an interruption), or there was a very long pause. Start a new block.
            groups.append({
                "label": label,
                "source_track_index": item.get("source_track_index"),
                "end_seconds": end,
                "transcript": transcript,
                "translation": translation,
            })

    # Format into final UI strings
    transcript_lines = [
        f"{group['label']}: {group['transcript']}" for group in groups
    ]
    translation_lines = [
        f"{group['label']}: {group['translation']}" for group in groups if group['translation']
    ]
    
    return "\n\n".join(transcript_lines), "\n\n".join(translation_lines)

def srt_timestamp(seconds):
    milliseconds = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_results(
    job_dir, records, metadata, quality_report, speaker_timeline,
    raw_speaker_timeline=None,
):
    timestamped_transcript, timestamped_translation = format_results(records)
    conversation_transcript, conversation_translation = format_conversation(records)
    (job_dir / "transcript.txt").write_text(
        conversation_transcript + "\n", encoding="utf-8"
    )
    (job_dir / "translation.txt").write_text(
        conversation_translation + "\n", encoding="utf-8"
    )
    (job_dir / "timestamped_transcript.txt").write_text(
        timestamped_transcript + "\n", encoding="utf-8"
    )
    (job_dir / "timestamped_translation.txt").write_text(
        timestamped_translation + "\n", encoding="utf-8"
    )
    (job_dir / "results.json").write_text(
        json.dumps({
            "metadata": metadata,
            "quality_report": quality_report,
            "speaker_timeline": speaker_timeline,
            "raw_speaker_timeline": (
                raw_speaker_timeline if raw_speaker_timeline is not None
                else speaker_timeline
            ),
            "segments": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    transcript_srt, translation_srt = [], []
    for index, item in enumerate(records, start=1):
        timing = f"{srt_timestamp(item['start_seconds'])} --> {srt_timestamp(item['end_seconds'])}"
        transcript_srt.append(
            f"{index}\n{timing}\n{item['speaker']}: {item['transcript']}\n"
        )
        translation_srt.append(
            f"{index}\n{timing}\n{item['speaker']}: {item.get('translation', '')}\n"
        )
    (job_dir / "transcript.srt").write_text("\n".join(transcript_srt), encoding="utf-8")
    (job_dir / "translation.srt").write_text("\n".join(translation_srt), encoding="utf-8")

    zip_path = job_dir / "accuracy_first_call_results.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(job_dir.iterdir()):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=path.name)
    return str(zip_path)


def _process_completed_media_impl(
    media_value,
    processing_profile="Balanced (recommended)",
    progress=gr.Progress(),
):
    if not media_value:
        raise gr.Error("Record or upload a completed audio/video file first.")
    started = time.time()
    progress(0.03, desc="Stage 1/2 · Preparing audio and inspecting channels...")
    source, job_dir, tracks, media_metadata = prepare_media(media_value)
    progress(0.10, desc="Stage 1/2 · Audio prepared; starting transcription...")

    selected_candidates, candidate_reports = [], []
    for index, track in enumerate(tracks, start=1):
        selected, report = transcribe_accuracy_first(
            track,
            processing_profile=processing_profile,
            progress=progress,
            track_index=index,
            track_count=len(tracks),
        )
        selected_candidates.append(selected)
        candidate_reports.append({
            "track": track["speaker"] or "mono mix",
            "selected_variant": selected["variant"],
            "selected_requested_language": selected["requested_language"],
            "selected_detected_language": selected["detected_language"],
            "selected_quality_score": selected["quality_score"],
            "selected_prompt_profile": selected.get("prompt_profile", "none"),
            "accent_context_retry_executed": any(
                item.get("prompt_profile") == "indian-english-hinglish-callcenter"
                for item in report
            ),
            "all_candidates": report,
        })

    raw_speaker_timeline = []
    speaker_timeline = []
    recognized_intervals = []
    
    # Matching strategy name from Step 5
    if media_metadata["channel_strategy"] == "separated_call_channels":
        records = []
        for track_index, (track, candidate) in enumerate(zip(tracks, selected_candidates)):
            records.extend(build_channel_turns(candidate, track["speaker"], track_index))
        records = mark_channel_overlaps(sorted(records, key=lambda item: item["start_seconds"]))
    else:
        track = tracks[0]
        selected_whisper = selected_candidates[0]
        
        # === CONDITIONAL NEMO ENHANCED AUDIO FALLBACK ===
        snr_db = float(track["raw_metrics"].get("approx_snr_db", 99.0))
        whisper_used_enhanced = (selected_whisper.get("variant") == "enhanced")
        
        nemo_target_path = track.get("diarization_path", track["raw_path"])
        
        if snr_db < 12.0 or whisper_used_enhanced or selected_whisper.get("quality_score", 1.0) < 0.20:
            enhanced_path = track["enhanced_path"]
            if not Path(enhanced_path).exists():
                progress(0.45, desc="Preparing enhanced audio track for NeMo diarization...")
                run_ffmpeg(track["raw_path"], enhanced_path, enhanced=True)
            
            nemo_target_path = enhanced_path
            media_metadata["nemo_audio_variant"] = "enhanced"
        else:
            media_metadata["nemo_audio_variant"] = "raw"

        progress(0.50, desc="Running NeMo Sortformer speaker diarization...")
        raw_speaker_timeline = run_nemo_diarization(nemo_target_path)
        speaker_timeline, recognized_intervals = gate_timeline_to_recognized_speech(
            raw_speaker_timeline, selected_whisper
        )
        records = build_mono_speaker_turns(selected_whisper, raw_speaker_timeline)

    if not records:
        raise gr.Error("Speech was detected, but no timestamped transcript was produced.")

    records, postprocessing_stats = postprocess_records(
        records, media_metadata["duration_seconds"]
    )
    if not records:
        raise gr.Error("Post-processing removed all invalid transcript records.")

    prepared_records = []
    for item in records:
        detected_code, _, detected_name = detect_text_language(item["transcript"], item.get("whisper_language"))
        prepared_records.append({
            **item,
            "detected_language": detected_code,
            "detected_language_name": detected_name,
            "translation_direction": "Pending Stage 2",
            "translation_model": "Not run",
            "translation": "",
        })
    records = prepared_records
    whisper_translation_fallback_passes = 0

    overlap_count = sum(bool(item.get("overlap_detected")) for item in records)
    uncertain_count = sum(
        bool(item.get("speaker_assignment_uncertain")) for item in records
    )
    warnings = []
    if media_metadata["channel_strategy"] == "separated_call_channels":
        warnings.append(
            "Two distinct channels were transcribed separately; simultaneous speech is retained."
        )
    else:
        warnings.append(
            "Mono diarization labels overlapping speaker activity, but a mono mixture may hide one speaker's words."
        )
        warnings.append(
            "Speaker labels outside recognized word-time regions were removed; the original recording clock was preserved."
        )
        if media_metadata["channels"] > 2:
            warnings.append(
                "The source has more than two channels, so all channels were downmixed for diarization instead of dropping any channel."
            )
    if overlap_count:
        warnings.append(f"Overlap was detected in {overlap_count} transcript turn(s); review those turns manually.")
    if uncertain_count:
        warnings.append(
            f"Speaker attribution was uncertain in {uncertain_count} turn(s); these labels were not forced."
        )
    context_correction_count = int(
        postprocessing_stats.get("context_correction_count", 0)
    )
    if context_correction_count:
        warnings.append(
            f"Context-aware post-processing applied {context_correction_count} high-confidence "
            "word/phrase correction(s); original ASR text is preserved in raw_transcript."
        )

    for track_number, track in enumerate(tracks, start=1):
        metrics = track["raw_metrics"]
        if metrics["approx_snr_db"] < 8.0:
            warnings.append(
                f"Track {track_number} has low approximate SNR ({metrics['approx_snr_db']:.1f} dB); review transcript accuracy."
            )
        if metrics["clipping_fraction"] > 0.005:
            warnings.append(
                f"Track {track_number} contains clipping; clipped speech cannot be restored by normalization."
            )

    duration_timeline = (
        raw_speaker_timeline
        if raw_speaker_timeline
        else [
            {
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "speaker": item["speaker"],
            }
            for item in records
        ]
    )
    speaker_durations = timeline_speaker_durations(duration_timeline)
    for speaker, duration in speaker_durations.items():
        if duration < 1.5:
            warnings.append(
                f"{speaker} has only {duration:.2f}s of speech evidence; it may be a brief speaker or background voice."
            )

    quality_report = {
        "model_accuracy_is_not_a_file_accuracy_guarantee": True,
        "channel_strategy": media_metadata["channel_strategy"],
        "source_channels": media_metadata["channels"],
        "channel_correlation": media_metadata.get("channel_correlation"),
        "channel_separation_diagnostics": media_metadata.get("channel_separation_diagnostics"),
        "crosstalk_suppression": media_metadata.get("crosstalk_suppression"),
        "preprocessing_policy": {
            "raw_candidate": "untouched isolated channel or 16 kHz mono resample with original timing preserved",
            "crosstalk_candidate": (
                "additional comparison only; it never replaces the untouched channel evidence"
            ),
            "enhanced_candidate": (
                "65–7600 Hz band-limit, adaptive FFT denoise, mild compression, "
                "dynamic normalization and peak limiting"
            ),
            "silence_removed": False,
            "speaker_identity_audio_enhanced": False,
        },
        "postprocessing_stats": postprocessing_stats,
        "track_audio_metrics": [track["raw_metrics"] for track in tracks],
        "diarization_audio_metrics": [
            track.get("diarization_metrics") for track in tracks
            if track.get("diarization_metrics") is not None
        ],
        "asr_candidate_selection": candidate_reports,
        "processing_profile": processing_profile,
        "asr_passes_executed": sum(
            len(report["all_candidates"]) for report in candidate_reports
        ),
        "accent_context_retry_count": sum(
            bool(report.get("accent_context_retry_executed"))
            for report in candidate_reports
        ),
        "context_correction_count": int(
            postprocessing_stats.get("context_correction_count", 0)
        ),
        "context_corrected_turn_count": int(
            postprocessing_stats.get("context_corrected_turns", 0)
        ),
        "whisper_translation_fallback_passes": whisper_translation_fallback_passes,
        "gpu_cleanup_policy": (
            "gc.collect, CUDA synchronize, empty_cache and ipc_collect after every request; "
            "models remain loaded"
        ),
        "translation_stage": "pending",
        "translation_models_used": [],
        "detected_speakers": sorted({item["speaker"] for item in records}),
        "speaker_time_seconds": speaker_durations,
        "raw_nemo_segment_count": len(raw_speaker_timeline),
        "speech_gated_nemo_segment_count": len(speaker_timeline),
        "recognized_speech_interval_count": len(recognized_intervals),
        "overlap_turn_count": overlap_count,
        "uncertain_speaker_turn_count": uncertain_count,
        "diarization_postprocessing": (
            None if media_metadata["channel_strategy"] == "separated_call_channels"
            else {
                "onset": 0.64,
                "offset": 0.74,
                "pad_onset": 0.06,
                "pad_offset": 0.0,
                "min_duration_on": 0.10,
                "min_duration_off": 0.15,
            }
        ),
        "warnings": warnings,
    }
    metadata = {
        **media_metadata,
        "duration_seconds": round(media_metadata["duration_seconds"], 3),
        "prepared_sample_rate": SAMPLE_RATE,
        "asr_model": ASR_MODEL_ID,
        "asr_implementation": "openai-whisper reference PyTorch",
        "diarization_model": (
            None if media_metadata["channel_strategy"] == "separated_call_channels"
            else DIARIZATION_MODEL_ID
        ),
        "translation_enabled": False,
        "translation_policy": (
            "Stage 2 translates the finalized speaker turns only after Stage 1 has returned. "
            "Optional turn-level Whisper audio recovery is never run automatically."
        ),
        "translation_output": "text_only_no_voice_synthesis",
        "manual_asr_chunking": False,
        "postprocessing_policy": (
            "timestamp clamping, invalid-record removal, overlapping near-duplicate removal, "
            "raw ASR preservation, auditable context/accent correction, "
            "conservative full-conversation speaker resolution, "
            "safe bridged-fragment reassignment and conservative same-speaker turn merging"
        ),
        "ui_presentation": "sequential speaker turns without timestamps",
        "transient_gpu_cache_cleared_after_request": True,
        "diarization_audio_policy": (
            "separate source channels; no NeMo required"
            if media_metadata["channel_strategy"] == "separated_call_channels"
            else f"complete speaker-preserving 16 kHz mono track ({media_metadata.get('nemo_audio_variant', 'raw')})"
        ),
    }
    transcript_text, _ = format_conversation(records)
    translation_text = "Stage 1 is complete. Click Step 2 to translate this finalized conversation."
    download = export_results(
        job_dir, records, metadata, quality_report, speaker_timeline,
        raw_speaker_timeline,
    )
    elapsed = time.time() - started
    progress(1.0, desc="Stage 1 complete · transcript and speaker labels are ready")
    status = (
        f"Stage 1 complete · {media_metadata['duration_seconds']:.1f}s · "
        f"{media_metadata['channel_strategy'].replace('_', ' ')} · "
        f"{processing_profile.split(' (')[0]} · "
        f"{len(records)} sequential speaker turn(s) · {overlap_count} overlap turn(s) · "
        f"{uncertain_count} uncertain speaker turn(s) · "
        f"processed in {elapsed:.1f}s. Review the transcript, then click Step 2 for translation."
    )
    stage_state = {
        "job_dir": str(job_dir),
        "tracks": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in track.items()
            }
            for track in tracks
        ],
        "records": records,
        "metadata": metadata,
        "quality_report": quality_report,
        "speaker_timeline": speaker_timeline,
        "raw_speaker_timeline": raw_speaker_timeline,
        "stage_1_elapsed_seconds": round(elapsed, 3),
    }
    return (
        transcript_text, translation_text, records, quality_report, download, status,
        stage_state, gr.update(interactive=True),
    )


def process_completed_media(
    media_value,
    processing_profile="Balanced (recommended)",
    progress=gr.Progress(),
):
    try:
        return _process_completed_media_impl(
            media_value,
            processing_profile=processing_profile,
            progress=progress,
        )
    finally:
        release_transient_gpu_memory()


def _translate_finalized_impl(
    stage_state,
    enable_audio_fallback=False,
    progress=gr.Progress(),
):
    if not stage_state or not stage_state.get("records"):
        raise gr.Error("Complete Step 1 transcription and speaker identification first.")

    started = time.time()
    records = [dict(item) for item in stage_state["records"]]
    tracks = [dict(track) for track in stage_state["tracks"]]
    metadata = dict(stage_state["metadata"])
    quality_report = dict(stage_state["quality_report"])
    speaker_timeline = list(stage_state.get("speaker_timeline") or [])
    raw_speaker_timeline = list(stage_state.get("raw_speaker_timeline") or speaker_timeline)
    job_dir = Path(stage_state["job_dir"])
    if not job_dir.exists():
        raise gr.Error("The prepared recording expired. Run Step 1 again.")

    progress(0.12, desc="Stage 2/2 · Reading finalized English/Hindi speaker turns...")
    with INFERENCE_LOCK:
        progress(0.28, desc="Stage 2/2 · Translating finalized text with IndicTrans2...")
        records, fallback_passes = translate_records_batch(
            records,
            tracks,
            enable_audio_fallback=bool(enable_audio_fallback),
        )

    elapsed = time.time() - started
    quality_report.update({
        "translation_stage": "complete",
        "translation_models_used": sorted({item["translation_model"] for item in records}),
        "translation_audio_fallback_requested": bool(enable_audio_fallback),
        "whisper_translation_fallback_passes": fallback_passes,
        "translation_review_turn_count": sum(
            bool(item.get("translation_review_required")) for item in records
        ),
        "translation_stage_elapsed_seconds": round(elapsed, 3),
    })
    metadata.update({
        "translation_enabled": True,
        "translation_policy": (
            "IndicTrans2 complete finalized speaker turns, with targeted turn-level Whisper recovery "
            "enabled only when explicitly requested"
            if enable_audio_fallback else
            "IndicTrans2 complete finalized speaker turns; targeted turn-level Whisper recovery disabled"
        ),
    })

    transcript_text, translation_text = format_conversation(records)
    download = export_results(
        job_dir, records, metadata, quality_report, speaker_timeline,
        raw_speaker_timeline,
    )
    updated_state = {
        **stage_state,
        "records": records,
        "metadata": metadata,
        "quality_report": quality_report,
        "translation_elapsed_seconds": round(elapsed, 3),
    }
    progress(1.0, desc="Stage 2 complete · translation is ready")
    status = (
        f"Stage 2 complete · {len(records)} finalized speaker turn(s) translated in {elapsed:.1f}s · "
        f"slow audio fallback passes: {fallback_passes}."
    )
    return (
        transcript_text, translation_text, records, quality_report, download, status,
        updated_state,
    )


def translate_finalized_media(
    stage_state,
    enable_audio_fallback=False,
    progress=gr.Progress(),
):
    try:
        return _translate_finalized_impl(
            stage_state,
            enable_audio_fallback=enable_audio_fallback,
            progress=progress,
        )
    finally:
        release_transient_gpu_memory()


def clear_results():
    return (
        "", "", [], {}, None, "Ready — complete Step 1 first.", {},
        gr.update(interactive=False),
    )

# ## Step 10 — Launch the two-stage recorded-media interface
# 
# Use one of three inputs: record audio, record video, or upload a completed call file. Run Step 1 for transcription plus speaker identification, review the result, then run Step 2 for English↔Hindi translation.


previous_demo = globals().get("demo")
if previous_demo is not None:
    try:
        previous_demo.close()
    except Exception:
        pass

with gr.Blocks(
    title="Accuracy-first recorded call transcription",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# Audio and Video Transcription and Translation\n"
        # "Record or upload a completed call. Whisper large-v3 transcribes English, Hindi and Hinglish; "
        # "the system preserves separate call channels when available and diarizes mono conversations.\n\n"
        # "**Step 1 returns the transcript and speaker labels before any translation begins.** "
        # "After reviewing them, Step 2 translates only the finalized turns. "
        # "Balanced ASR compares auto, English and Hindi whole-file decodes, while preserving untouched stereo call legs. "
        # "When the transcript shows strong Indian-English/Hinglish call-center context, one vocabulary-guided retry is added. "
        # "You may submit one additional file while processing; it will wait in the bounded queue. "
        # "Temporary CUDA cache is cleared after every completed or failed request."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("🎙 Record audio"):
                    audio_input = gr.Audio(
                        sources=["microphone"], type="filepath",
                        label="Completed microphone recording",
                    )
                    audio_button = gr.Button(
                        "Step 1 · Transcribe + identify speakers", variant="primary"
                    )

                with gr.Tab("🎥 Record video"):
                    video_input = gr.Video(
                         sources=["webcam"],
                         format=None,
                         include_audio=True,
                         # webcam_options={
                         #    "mirror": False,
                         #    "constraints": {
                         #        "video": True,
                         #        "audio": True,
                         # },
                     # },
                    )
                    video_button = gr.Button(
                        "Step 1 · Transcribe + identify speakers", variant="primary"
                    )

                with gr.Tab("📁 Upload call"):
                    file_input = gr.File(
                        file_types=["audio", "video"], type="filepath",
                        label="Upload an audio or video file (maximum 60 minutes)",
                    )
                    file_button = gr.Button(
                        "Step 1 · Transcribe + identify speakers", variant="primary"
                    )

            translate_button = gr.Button(
                "Step 2 · Translate finalized conversation",
                variant="secondary",
                interactive=False,
            )
            # translation_audio_fallback = gr.Checkbox(
            #     value=False,
            #     label="Use targeted original-audio recovery for failed Hindi → English turns",
            #     info="Leave off for normal use. Turning it on decodes only the finalized turns that failed text translation.",
            # )
            processing_profile = gr.Radio(
                choices=[
                    "Balanced (recommended)",
                    "Maximum accuracy (slow)",
                ],
                value="Balanced (recommended)",
                label="Processing profile",
                # info="Balanced compares 3–5 passes per track; Maximum compares 6 passes, or 9 for separated stereo call legs.",
            )
            cancel_button = gr.Button("Cancel waiting requests", variant="stop")
            clear_button = gr.Button("Clear results")
            status = gr.Textbox(
                value="Ready — complete Step 1 first.",
                label="Processing status", lines=3, interactive=False,
            )
            download = gr.File(label="Download transcript, translation, JSON and subtitles")

        with gr.Column(scale=2):
            transcript = gr.Textbox(
                label="Context-corrected speaker conversation", lines=14, interactive=False,
            )
            gr.Markdown(
                "Context corrections require strong domain evidence. "
                "Structured records preserve raw_transcript and list every contextual_correction."
            )
            translation = gr.Textbox(
                label="Sequential English ↔ Hindi conversation", lines=12, interactive=False,
            )

    with gr.Accordion("Quality and overlap report", open=False):
        quality_report = gr.JSON(label="Selected ASR pass, audio metrics and review warnings")

    with gr.Accordion("Structured timestamped records", open=False):
        records = gr.JSON(label="Speaker, overlap, language, confidence and text")

    stage_state = gr.State({})

    stage_outputs = [
        transcript, translation, records, quality_report, download, status, stage_state,
    ]
    step_1_outputs = stage_outputs + [translate_button]
    submit_events = []
    for button, media_input in (
        (audio_button, audio_input),
        (video_button, video_input),
        (file_button, file_input),
    ):
        submit_events.append(button.click(
            process_completed_media,
            inputs=[media_input, processing_profile],
            outputs=step_1_outputs,
            concurrency_limit=1,
            concurrency_id="accuracy_first_gpu",
            trigger_mode="multiple",
        ))

    translation_event = translate_button.click(
        translate_finalized_media,
        inputs=[stage_state, translation_audio_fallback],
        outputs=stage_outputs,
        concurrency_limit=1,
        concurrency_id="accuracy_first_gpu",
        trigger_mode="once",
    )
    cancel_button.click(
        fn=None, cancels=submit_events + [translation_event], queue=False
    )
    clear_button.click(clear_results, outputs=step_1_outputs, queue=False)

demo.queue(default_concurrency_limit=1, max_size=2, status_update_rate="auto")


import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

demo.launch(
    share=True,
    debug=False,
    show_error=True,
    inline=False,
    show_api=False,
    allowed_paths=[str(OUTPUT_ROOT)],  # Allows Gradio to serve files from /content/accuracy_outputs
)

import shutil
from pathlib import Path

# Source file path in /content
src = Path("/content/accuracy_outputs/call-20260821-065512-b2a571/audio/mono_enhanced.wav")
# Destination path in Kaggle working direcrictory
dst = Path("/kaggle/working/mono_enhanced.wav")

if src.exists():
    shutil.copy(src, dst)
    print("✅ File successfully copied to /kaggle/working/mono_enhanced.wav")
else:
    print("❌ Source file not found. Check if the job ID directory exists.")

# ## Final test checklist
# 
# 1. Select a GPU runtime and run all cells in order.
# 2. First test a clear 20–30 second English call, then Hindi, then Hinglish.
# 3. Test an uploaded stereo call where the agent and customer occupy different channels. Confirm the quality report says `separate_channels`.
# 4. Test a mono noisy call. Confirm `[overlap detected]` and `[speaker uncertain]` are shown instead of confident but unsupported labels.
# 5. Test background music/non-speech and a brief third voice. Check `speaker_time_seconds`, `approx_snr_db`, and the quality warnings.
# 6. Paste a human reference transcript and record WER/CER. For diarization, also compare speaker turns against a human RTTM/reference annotation.
# 7. Confirm Step 1 returns the transcript and speaker labels before Step 2 translation is started.
# 8. Run Step 2 with slow audio recovery off; enable it only to investigate a failed Hindi→English translation.
# 9. Download the ZIP and check TXT, JSON and SRT outputs.
# 10. Confirm the UI reads `Speaker 1`, then `Speaker 2`, then `Speaker 1` again when that speaker returns; timestamps should appear only in the timestamped TXT, JSON and SRT files.
# 
# ### Before a supervisor demo
# 
# - Use one known-good English/Hindi/Hinglish file that has already completed successfully.
# - Keep a human reference ready so the accuracy metric is reproducible.
# - Explain that stereo channel separation solves overlap much better than mono diarization.
# - Do not claim one public benchmark as the accuracy of this call-center system. Report WER/CER from your own held-out calls.
# 
# ### Production limits
# 
# - Up to four diarized speakers; call-center calls are normally two.
# - Mono overlap is marked for review but both hidden utterances may not be recoverable.
# - Speaker numbers identify voices only within one recording; they are not verified identities.
# - Browser/Colab recording is a prototype. Production uploads need authenticated storage, a durable queue, encryption, retention controls and human review for low-confidence/overlapped turns.
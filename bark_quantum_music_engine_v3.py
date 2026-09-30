# Bark Quantum Music Engine v2
# Extracted from Bark_Quantum_Music_Kernel_v2.ipynb
# Stateful long-form Bark music generation with token memory, candidate search, and bar-locked control.

from __future__ import annotations

import gc
import contextlib
import json
import math
import random
import time
import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io.wavfile import write as scipy_write_wav
from scipy.signal import get_window

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from IPython.display import Audio, display
except Exception:
    Audio = None
    display = print

BARK_AVAILABLE = False
try:
    import bark
    import bark.generation as bark_generation
    BARK_AVAILABLE = True
except Exception as exc:
    bark = None
    bark_generation = None
    print("Bark not imported yet:", exc)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def stable_hash(*parts: str, bits: int = 64) -> int:
    h = hashlib.blake2b("||".join(parts).encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(h, "big")


@dataclass
class KernelConfig:
    seed: int = 1337
    candidates_per_segment: int = 3
    segment_bars: int = 2
    max_segment_seconds: float = 12.0
    crossfade_ms: float = 45.0

    # Base Bark sampling.
    semantic_temp: float = 0.58
    coarse_temp: float = 0.60
    fine_temp: Optional[float] = 0.45
    semantic_top_k: Optional[int] = 90
    semantic_top_p: Optional[float] = 0.92
    coarse_top_k: Optional[int] = 90
    coarse_top_p: Optional[float] = 0.92

    # Music-aware semantic-loop bias strengths. Keep these modest.
    token_prior_strength: float = 0.40
    bigram_strength: float = 0.48
    phase_strength: float = 0.35
    harmony_strength: float = 0.28
    motif_strength: float = 0.35
    max_logit_bias: float = 1.25

    # Multi-timescale Bark history budget, in semantic tokens.
    macro_anchor_tokens: int = 36
    episodic_tokens: int = 48
    recent_tokens: int = 120

    # Memory sizes.
    phase_bins: int = 16
    harmonic_slots: int = 12
    max_bigram_followers: int = 48
    meso_segments: int = 12
    episodic_segments: int = 8

    # Candidate score weights.
    w_boundary: float = 0.90
    w_timbre: float = 0.85
    w_pulse: float = 0.75
    w_harmony: float = 0.90
    w_motif: float = 0.70
    w_novelty: float = 0.35
    w_level: float = 0.25

    # Optional CLAP anti-vocal mode.
    use_clap: bool = False
    instrumental_only: bool = True

    # Checkpoints / output.
    output_dir: str = "bark_quantum_music_run"
    keep_chunk_wavs: bool = True
    keep_history_npz: bool = True

    # Hardening. The custom sampler intentionally reaches into Bark internals.
    strict_bark_compat: bool = True
    allow_legacy_checkpoint_pickle: bool = True


CFG = KernelConfig()
seed_everything(CFG.seed)
CFG


BARK_CUSTOM_REQUIRED = (
    "models", "models_devices", "_tokenize", "_normalize_whitespace",
    "_load_history_prompt", "_inference_mode", "SEMANTIC_VOCAB_SIZE",
    "TEXT_ENCODING_OFFSET", "TEXT_PAD_TOKEN", "SEMANTIC_PAD_TOKEN",
    "SEMANTIC_INFER_TOKEN", "SEMANTIC_RATE_HZ",
)


def bark_compat_report() -> Dict[str, Any]:
    if not BARK_AVAILABLE:
        return {"bark_available": False, "custom_sampler_compatible": False, "missing": list(BARK_CUSTOM_REQUIRED)}
    missing=[name for name in BARK_CUSTOM_REQUIRED if not hasattr(bark_generation,name)]
    return {
        "bark_available": True,
        "custom_sampler_compatible": not missing,
        "missing": missing,
        "semantic_rate_hz": float(getattr(bark_generation,"SEMANTIC_RATE_HZ",49.9)),
        "semantic_vocab_size": int(getattr(bark_generation,"SEMANTIC_VOCAB_SIZE",10_000)),
    }


@contextlib.contextmanager
def legacy_bark_checkpoint_loading(enabled: bool=True):
    """Scoped PyTorch 2.6+ compatibility for the official legacy Bark checkpoints.

    `weights_only=False` permits pickle execution. Use only with a checkpoint source you trust.
    """
    if not enabled or torch is None:
        yield
        return
    original_load=torch.load
    def _compat_load(*args,**kwargs):
        kwargs.setdefault("weights_only",False)
        return original_load(*args,**kwargs)
    torch.load=_compat_load
    try:
        yield
    finally:
        torch.load=original_load


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


def note_to_pc(note: str) -> int:
    n = note.strip().upper().replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    if n not in NOTE_NAMES:
        raise ValueError(f"Unsupported root note: {note}")
    return NOTE_NAMES.index(n)


def chord_chroma(root_pc: int, quality: str = "minor") -> np.ndarray:
    quality = quality.lower()
    intervals = {
        "major": [0, 4, 7], "minor": [0, 3, 7], "dim": [0, 3, 6],
        "sus2": [0, 2, 7], "sus4": [0, 5, 7], "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10], "dom7": [0, 4, 7, 10],
    }.get(quality, [0, 3, 7])
    x = np.zeros(12, dtype=np.float64)
    for i in intervals:
        x[(root_pc + i) % 12] = 1.0
    return x / (np.linalg.norm(x) + 1e-12)


@dataclass
class SongDNA:
    title: str = "Quantum Bloom"
    bpm: float = 128.0
    meter_numerator: int = 4
    meter_denominator: int = 4
    root: str = "D"
    mode: str = "minor"
    style: str = "experimental electronic, hyper-detailed, organic-digital, high dynamic range"
    instruments: str = "membrane kick, ceramic clicks, glass FM plucks, pure sine sub, prepared piano grains"
    production: str = "dry close transients, enormous rear depth, clean mono sub, no clipping"
    mood: str = "tense, luminous, strange, emotional"
    vocal_mode: str = "instrumental"  # instrumental | singing
    voice_preset: Optional[str] = None
    chord_degrees: Tuple[int, ...] = (1, 6, 3, 7)  # scale degrees 1..7
    chord_qualities: Tuple[str, ...] = ("minor", "major", "major", "major")

    @property
    def root_pc(self) -> int:
        return note_to_pc(self.root)

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / float(self.bpm)

    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * self.meter_numerator

    def scale_pcs(self) -> List[int]:
        ints = MODE_INTERVALS.get(self.mode.lower(), MODE_INTERVALS["minor"])
        return [(self.root_pc + i) % 12 for i in ints]

    def chord_for_bar(self, bar_index: int) -> Tuple[int, str, np.ndarray, str]:
        pcs = self.scale_pcs()
        idx = bar_index % len(self.chord_degrees)
        degree = int(self.chord_degrees[idx])
        root_pc = pcs[(degree - 1) % len(pcs)]
        quality = self.chord_qualities[idx % len(self.chord_qualities)]
        return root_pc, quality, chord_chroma(root_pc, quality), f"{NOTE_NAMES[root_pc]} {quality}"

    def harmonic_slot_for_bar(self, bar_index: int) -> int:
        """Twelve-state harmonic memory slot keyed to the actual chord root pitch class."""
        root_pc, _, _, _ = self.chord_for_bar(int(bar_index))
        return int(root_pc)

    def core_prompt(self) -> str:
        no_voice = "instrumental only, no voice, no spoken words" if self.vocal_mode == "instrumental" else "sung music, preserve singer identity"
        return (
            f"[music] {self.style}; {self.bpm:.1f} BPM; {self.root} {self.mode}; "
            f"{self.instruments}; {self.production}; {self.mood}; {no_voice}"
        )


@dataclass
class Section:
    name: str
    bars: int
    direction: str
    energy: float = 0.5
    mutation: float = 0.25
    lyrics: str = ""


@dataclass
class Segment:
    index: int
    section_name: str
    bar_start: int
    bars: int
    seconds: float
    direction: str
    energy: float
    mutation: float
    lyrics: str = ""
    is_section_start: bool = False
    is_section_end: bool = False


@dataclass
class SongPlan:
    dna: SongDNA
    sections: List[Section]
    segment_bars: int = 2
    max_segment_seconds: float = 12.0

    def segments(self) -> List[Segment]:
        out: List[Segment] = []
        global_bar = 0
        idx = 0
        for sec in self.sections:
            # shrink the chunk size automatically when BPM is slow.
            bars_per = max(1, int(self.segment_bars))
            while bars_per * self.dna.seconds_per_bar > self.max_segment_seconds and bars_per > 1:
                bars_per -= 1
            remaining = sec.bars
            local_bar = 0
            while remaining > 0:
                n = min(bars_per, remaining)
                out.append(Segment(
                    index=idx,
                    section_name=sec.name,
                    bar_start=global_bar,
                    bars=n,
                    seconds=n * self.dna.seconds_per_bar,
                    direction=sec.direction,
                    energy=float(np.clip(sec.energy, 0, 1)),
                    mutation=float(np.clip(sec.mutation, 0, 1)),
                    lyrics=sec.lyrics,
                    is_section_start=(local_bar == 0),
                    is_section_end=(remaining == n),
                ))
                idx += 1
                local_bar += n
                global_bar += n
                remaining -= n
        return out


def peak_normalize(audio: np.ndarray, peak: float = 0.985) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > peak and m > 0:
        x = x * (peak / m)
    return x.astype(np.float32, copy=False)


def write_wav(path: str | Path, sr: int, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = peak_normalize(audio)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    scipy_write_wav(path, int(sr), pcm)


def equal_power_join(a: np.ndarray, b: np.ndarray, sr: int, ms: float = 45.0) -> np.ndarray:
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    n = min(len(a), len(b), max(0, int(sr * ms / 1000.0)))
    if n <= 1:
        return np.concatenate([a, b])
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    mixed = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mixed, b[n:]])


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def frame_audio(x: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    x = np.asarray(x, np.float64).reshape(-1)
    if len(x) < frame:
        x = np.pad(x, (0, frame-len(x)))
    n = 1 + (len(x)-frame)//hop
    idx = np.arange(frame)[None, :] + hop*np.arange(n)[:, None]
    return x[idx]


def stft_power(x: np.ndarray, sr: int, frame: int = 2048, hop: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    frames = frame_audio(x, frame, hop)
    win = get_window("hann", frame, fftbins=True)
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    p = np.abs(spec)**2
    f = np.fft.rfftfreq(frame, 1.0/sr)
    return p, f


def chroma_vector(x: np.ndarray, sr: int) -> np.ndarray:
    p, f = stft_power(x, sr)
    mask = (f >= 45.0) & (f <= 5000.0)
    f2 = f[mask]
    energy = p[:, mask].sum(axis=0)
    midi = np.rint(69 + 12*np.log2(np.maximum(f2, 1e-6)/440.0)).astype(int)
    chroma = np.zeros(12, dtype=np.float64)
    for pc in range(12):
        chroma[pc] = energy[(midi % 12) == pc].sum()
    chroma = np.log1p(chroma)
    return chroma / (np.linalg.norm(chroma) + 1e-12)


def spectral_signature(x: np.ndarray, sr: int, bands: int = 24) -> np.ndarray:
    p, f = stft_power(x, sr)
    e = p.mean(axis=0) + 1e-12
    lo, hi = 45.0, min(sr/2.0, 12000.0)
    edges = np.geomspace(lo, hi, bands+1)
    vals=[]
    for a,b in zip(edges[:-1], edges[1:]):
        m=(f>=a)&(f<b)
        vals.append(np.log1p(e[m].sum()))
    v=np.asarray(vals, np.float64)
    return (v-v.mean())/(v.std()+1e-8)


def onset_envelope(x: np.ndarray, sr: int, frame: int = 1024, hop: int = 256) -> Tuple[np.ndarray,float]:
    frames=frame_audio(x,frame,hop)
    e=np.sqrt(np.mean(frames*frames,axis=1)+1e-12)
    env=np.maximum(0,np.diff(e,prepend=e[0]))
    env=env/(np.linalg.norm(env)+1e-12)
    return env, hop/sr


def pulse_strength(x: np.ndarray, sr: int, bpm: float) -> float:
    env, dt = onset_envelope(x,sr)
    lag=max(1,int(round((60.0/bpm)/dt)))
    if len(env)<=lag+2:
        return 0.0
    a,b=env[:-lag],env[lag:]
    return max(0.0, cosine(a,b))


def boundary_similarity(prev: Optional[np.ndarray], cur: np.ndarray, sr: int, seconds: float=1.0) -> float:
    if prev is None or len(prev)==0:
        return 0.5
    n=int(sr*seconds)
    a=np.asarray(prev[-n:],np.float32)
    b=np.asarray(cur[:n],np.float32)
    return 0.5*cosine(spectral_signature(a,sr),spectral_signature(b,sr)) + 0.5*(1.0-min(1.0,abs(rms(a)-rms(b))*4.0))


def audio_state_vector(x: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    chroma=chroma_vector(x,sr)
    spec=spectral_signature(x,sr,24)
    level=np.array([np.tanh(4*rms(x)), pulse_strength(x,sr,bpm)],dtype=np.float64)
    v=np.concatenate([chroma,spec,level])
    return v/(np.linalg.norm(v)+1e-12)


def entropy_bits(p: np.ndarray) -> float:
    p=np.asarray(p,np.float64).reshape(-1)
    p=np.clip(p,0,None)
    p=p/(p.sum()+1e-12)
    nz=p[p>1e-15]
    return float(-np.sum(nz*np.log2(nz)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p=np.asarray(p,np.float64); q=np.asarray(q,np.float64)
    p=np.clip(p,0,None); q=np.clip(q,0,None)
    p/=p.sum()+1e-12; q/=q.sum()+1e-12
    m=0.5*(p+q)
    def kl(a,b):
        mask=a>1e-15
        return float(np.sum(a[mask]*np.log2((a[mask]+1e-15)/(b[mask]+1e-15))))
    return 0.5*kl(p,m)+0.5*kl(q,m)


def robust_z(value: float, series: Sequence[float]) -> float:
    s=np.asarray(series,np.float64)
    if len(s)<4: return 0.0
    med=float(np.median(s)); mad=float(np.median(np.abs(s-med)))
    if mad<1e-12:
        sd=float(np.std(s)); return 0.0 if sd<1e-12 else (value-med)/sd
    return 0.67448975*(value-med)/mad


@dataclass
class WorldlinePoint:
    segment_index: int
    state: np.ndarray
    quantum_probs: np.ndarray
    harmony_score: float
    pulse_score: float
    motif_score: float
    novelty: float


class MusicalWorldline:
    """Tracks movement through a compact musical feature manifold."""
    def __init__(self, maxlen: int=64):
        self.points: Deque[WorldlinePoint]=deque(maxlen=maxlen)

    def append(self, point: WorldlinePoint) -> None:
        self.points.append(point)

    def kinematics(self) -> Dict[str,float]:
        if len(self.points)<2:
            return {"velocity":0.0,"acceleration":0.0,"curvature":0.0,"q_js":0.0,"stagnation":0.0}
        xs=[p.state for p in self.points]
        v=np.linalg.norm(xs[-1]-xs[-2])
        a=0.0; c=0.0
        if len(xs)>=3:
            v0=xs[-2]-xs[-3]; v1=xs[-1]-xs[-2]
            a=float(np.linalg.norm(v1-v0))
            denom=(np.linalg.norm(v0)*np.linalg.norm(v1)+1e-12)
            c=float(1.0-np.clip(np.dot(v0,v1)/denom,-1,1))
        qjs=js_divergence(self.points[-1].quantum_probs,self.points[-2].quantum_probs)
        recent=[np.linalg.norm(xs[i]-xs[i-1]) for i in range(1,len(xs))][-6:]
        stagnation=float(np.clip(1.0-(np.mean(recent) if recent else 0.0)*3.5,0,1))
        return {"velocity":float(v),"acceleration":a,"curvature":c,"q_js":qjs,"stagnation":stagnation}


def _single_qubit_gate(nq: int, wire: int, gate: np.ndarray) -> np.ndarray:
    out=np.array([[1.0+0j]])
    for w in range(nq):
        out=np.kron(out, gate if w==wire else np.eye(2,dtype=complex))
    return out


def _ry(theta: float) -> np.ndarray:
    c,s=np.cos(theta/2),np.sin(theta/2)
    return np.array([[c,-s],[s,c]],dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j*theta),0],[0,np.exp(0.5j*theta)]],dtype=complex)


def _controlled_gate(nq: int, control: int, target: int, z_only: bool=False) -> np.ndarray:
    size=2**nq
    U=np.zeros((size,size),dtype=complex)
    for i in range(size):
        bits=[(i>>(nq-1-w))&1 for w in range(nq)]
        j=i
        amp=1.0+0j
        if bits[control]:
            if z_only:
                if bits[target]: amp=-1.0
            else:
                bits[target]^=1
                j=0
                for w,b in enumerate(bits): j|=(b<<(nq-1-w))
        U[j,i]=amp
    return U


class QuantumMusicalController:
    """
    Inputs are normalized feature-space measurements:
      drift, stagnation, harmony_error, pulse_error, token_entropy.
    Outputs are adaptive control values in [0,1].
    """
    NQ=5
    def __init__(self, layers: int=3):
        self.layers=layers
        self.cnot=[_controlled_gate(self.NQ,i,(i+1)%self.NQ,False) for i in range(self.NQ)]
        self.cz=[_controlled_gate(self.NQ,0,2,True),_controlled_gate(self.NQ,1,3,True),_controlled_gate(self.NQ,2,4,True)]

    def run(self, features: Sequence[float]) -> Dict[str,Any]:
        f=np.clip(np.asarray(features,np.float64),0,1)
        if len(f)!=self.NQ: raise ValueError("Quantum controller expects five features")
        state=np.zeros(2**self.NQ,dtype=complex); state[0]=1.0
        for layer in range(self.layers):
            mult=1.0+0.17*layer
            for w,x in enumerate(f):
                # magnitude + centered phase, mirroring robust magnitude/deviation encoding.
                state=_single_qubit_gate(self.NQ,w,_ry(np.pi*x*mult))@state
                state=_single_qubit_gate(self.NQ,w,_rz(np.pi*(x-0.5)/(layer+1)))@state
            if layer%2==0:
                for U in self.cnot: state=U@state
            else:
                for U in self.cz: state=U@state
        probs=np.abs(state)**2; probs/=probs.sum()+1e-15
        # Pauli-Z expectations from probability marginals.
        z=[]
        for w in range(self.NQ):
            ex=0.0
            for i,p in enumerate(probs):
                bit=(i>>(self.NQ-1-w))&1
                ex += p*(1.0 if bit==0 else -1.0)
            z.append(float(ex))
        c=[float(np.clip((1-v)/2,0,1)) for v in z]
        return {
            "probabilities": probs,
            "entropy_bits": entropy_bits(probs),
            "pauli_z": z,
            "exploration": c[0],
            "motif_pull": c[1],
            "beat_lock": c[2],
            "harmony_lock": c[3],
            "memory_mix": c[4],
        }

QCTRL=QuantumMusicalController()
QCTRL.run([0.2,0.4,0.1,0.3,0.5])


class PersistentTokenMemory:
    def __init__(self, vocab_size: int=10_000, phase_bins: int=16, harmonic_slots: int=12, max_followers: int=48):
        self.vocab_size=int(vocab_size)
        self.phase_bins=int(phase_bins)
        self.harmonic_slots=int(harmonic_slots)
        self.unigram=np.zeros(self.vocab_size,dtype=np.float64)
        self.phase_counts=np.zeros((self.phase_bins,self.vocab_size),dtype=np.float32)
        self.harmony_counts=np.zeros((self.harmonic_slots,self.vocab_size),dtype=np.float32)
        self.bigram: Dict[int,Counter]=defaultdict(Counter)
        self.max_followers=max_followers
        self.total=0

    def observe(self, semantic: np.ndarray, start_seconds: float, semantic_rate: float, bpm: float, harmonic_slot_fn: Callable[[float],int]) -> None:
        sem=np.asarray(semantic,dtype=np.int64).reshape(-1)
        if sem.size==0: return
        spb=60.0/bpm
        prev=None
        for i,t in enumerate(sem):
            if not (0<=t<self.vocab_size): continue
            ts=start_seconds+i/semantic_rate
            phase=((ts/spb)%1.0)
            pb=int(phase*self.phase_bins)%self.phase_bins
            hs=int(harmonic_slot_fn(ts))%self.harmonic_slots
            self.unigram[t]+=1.0
            self.phase_counts[pb,t]+=1.0
            self.harmony_counts[hs,t]+=1.0
            if prev is not None:
                c=self.bigram[int(prev)]
                c[int(t)]+=1
                if len(c)>self.max_followers*2:
                    self.bigram[int(prev)]=Counter(dict(c.most_common(self.max_followers)))
            prev=int(t); self.total+=1

    def token_entropy(self) -> float:
        if self.total<=0: return 1.0
        p=self.unigram/(self.unigram.sum()+1e-12)
        h=entropy_bits(p)
        return float(h/max(1e-12,math.log2(self.vocab_size)))

    @staticmethod
    def _log_prior(counts: np.ndarray) -> np.ndarray:
        p=(counts+0.1)/(float(np.sum(counts))+0.1*len(counts))
        lp=np.log(p+1e-12)
        lp-=np.mean(lp)
        sd=np.std(lp)+1e-8
        return lp/sd

    def logit_bias(self, prev_token: Optional[int], phase_bin: int, harmonic_slot: int, weights: Dict[str,float], cfg: KernelConfig) -> np.ndarray:
        if self.total<32:
            return np.zeros(self.vocab_size,dtype=np.float32)
        bias=cfg.token_prior_strength*self._log_prior(self.unigram)
        phase=self.phase_counts[phase_bin%self.phase_bins]
        if phase.sum()>4:
            bias += cfg.phase_strength*weights.get("beat_lock",0.5)*self._log_prior(phase)
        harmony=self.harmony_counts[harmonic_slot%self.harmonic_slots]
        if harmony.sum()>4:
            bias += cfg.harmony_strength*weights.get("harmony_lock",0.5)*self._log_prior(harmony)
        if prev_token is not None and prev_token in self.bigram and self.bigram[prev_token]:
            follower=np.zeros(self.vocab_size,dtype=np.float64)
            for tok,c in self.bigram[prev_token].most_common(self.max_followers): follower[tok]=c
            bias += cfg.bigram_strength*weights.get("motif_pull",0.5)*self._log_prior(follower)
        return np.clip(bias,-cfg.max_logit_bias,cfg.max_logit_bias).astype(np.float32)

    def fingerprint(self, semantic: np.ndarray, bins: int=32) -> np.ndarray:
        out=np.zeros(bins,dtype=np.float64)
        for t in np.asarray(semantic,dtype=np.int64).reshape(-1):
            out[(int(t)*2654435761 % 2**32)%bins]+=1.0
        return out/(np.linalg.norm(out)+1e-12)


@dataclass
class SegmentMemory:
    index: int
    bar_start: int
    start_seconds: float
    history: Dict[str,np.ndarray]
    audio_state: np.ndarray
    chroma: np.ndarray
    token_fp: np.ndarray
    score: float
    q_probs: np.ndarray
    harmony_slot: int


class RollingMusicMemory:
    def __init__(self, cfg: KernelConfig, token_memory: PersistentTokenMemory):
        self.cfg=cfg
        self.token_memory=token_memory
        self.recent_history: Optional[Dict[str,np.ndarray]]=None
        self.anchor_history: Optional[Dict[str,np.ndarray]]=None
        self.meso: Deque[SegmentMemory]=deque(maxlen=cfg.meso_segments)
        self.episodic: List[SegmentMemory]=[]
        self.macro_state: Optional[np.ndarray]=None
        self.macro_chroma: Optional[np.ndarray]=None
        self.macro_token_fp: Optional[np.ndarray]=None
        self.worldline=MusicalWorldline(maxlen=64)

    def commit(self, m: SegmentMemory, semantic_rate: float, bpm: float, harmonic_slot_fn: Callable[[float],int]) -> None:
        # Compare against the identity that existed *before* accepting this segment.
        previous_macro=None if self.macro_state is None else self.macro_state.copy()
        novelty=0.5 if previous_macro is None else 1.0-max(0.0,cosine(m.audio_state,previous_macro))

        self.recent_history=m.history
        if self.anchor_history is None:
            self.anchor_history=m.history
        alpha=0.12
        self.macro_state=m.audio_state.copy() if self.macro_state is None else (1-alpha)*self.macro_state+alpha*m.audio_state
        self.macro_chroma=m.chroma.copy() if self.macro_chroma is None else (1-alpha)*self.macro_chroma+alpha*m.chroma
        self.macro_token_fp=m.token_fp.copy() if self.macro_token_fp is None else (1-alpha)*self.macro_token_fp+alpha*m.token_fp
        self.meso.append(m)
        self.episodic=sorted(self.episodic+[m],key=lambda x:x.score,reverse=True)[:self.cfg.episodic_segments]
        self.token_memory.observe(m.history["semantic_prompt"],m.start_seconds,semantic_rate,bpm,harmonic_slot_fn)
        self.worldline.append(WorldlinePoint(m.index,m.audio_state,m.q_probs,0.0,0.0,0.0,novelty))

    def retrieve_exemplar(self, target_chroma: np.ndarray, target_state: Optional[np.ndarray]=None) -> Optional[SegmentMemory]:
        pool=list(self.meso)+list(self.episodic)
        if not pool: return None
        best=None; best_score=-1e9
        for m in pool:
            s=0.75*cosine(m.chroma,target_chroma)
            if target_state is not None: s+=0.25*cosine(m.audio_state,target_state)
            s+=0.05*m.score
            if s>best_score: best,best_score=m,s
        return best


def _history_arrays(h: Dict[str,np.ndarray]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    return np.asarray(h["semantic_prompt"]),np.asarray(h["coarse_prompt"]),np.asarray(h["fine_prompt"])


def tail_history(h: Dict[str,np.ndarray], semantic_tokens: int) -> Dict[str,np.ndarray]:
    sem,coarse,fine=_history_arrays(h)
    n=max(1,min(int(semantic_tokens),len(sem)))
    ratio=coarse.shape[1]/max(1,len(sem))
    frames=max(1,min(coarse.shape[1],fine.shape[1],int(round(n*ratio))))
    return {"semantic_prompt":sem[-n:].copy(),"coarse_prompt":coarse[:,-frames:].copy(),"fine_prompt":fine[:,-frames:].copy()}


def fuse_many_histories(parts: Sequence[Tuple[Optional[Dict[str,np.ndarray]],int]], expected_ratio: float=1.5) -> Optional[Dict[str,np.ndarray]]:
    chunks=[]
    for h,n in parts:
        if h is not None and n>0: chunks.append(tail_history(h,n))
    if not chunks: return None
    sem=np.concatenate([x["semantic_prompt"] for x in chunks])
    coarse=np.concatenate([x["coarse_prompt"] for x in chunks],axis=1)
    fine=np.concatenate([x["fine_prompt"] for x in chunks],axis=1)
    # Bark's coarse history expects roughly 1.5 coarse frames per semantic token.
    target=max(1,int(round(len(sem)*expected_ratio)))
    target=min(target,coarse.shape[1],fine.shape[1])
    # If concatenated histories differ slightly in native ratios, crop from the oldest side.
    coarse=coarse[:,-target:]
    fine=fine[:,-target:]
    max_sem=max(1,int(round(target/expected_ratio)))
    sem=sem[-max_sem:]
    return {"semantic_prompt":sem.astype(np.int32,copy=False),"coarse_prompt":coarse.astype(np.int32,copy=False),"fine_prompt":fine.astype(np.int32,copy=False)}


def _sample_logits(logits, temp: float, top_k: Optional[int], top_p: Optional[float]):
    scores=logits/max(1e-5,float(temp))
    if top_k is not None and 0<top_k<scores.shape[-1]:
        kth=torch.topk(scores,int(top_k)).values[-1]
        scores=torch.where(scores<kth,torch.full_like(scores,-float("inf")),scores)
    if top_p is not None and 0.0<top_p<1.0:
        sorted_scores,sorted_idx=torch.sort(scores,descending=True)
        probs=torch.softmax(sorted_scores,dim=-1)
        cumulative=torch.cumsum(probs,dim=-1)
        remove=cumulative>float(top_p)
        if remove.numel()>1:
            remove[1:]=remove[:-1].clone(); remove[0]=False
        sorted_scores[remove]=-float("inf")
        filtered=torch.full_like(scores,-float("inf"))
        filtered.scatter_(0,sorted_idx,sorted_scores)
        scores=filtered
    probs=torch.softmax(scores,dim=-1)
    return torch.multinomial(probs,1).item()


def music_generate_text_semantic(
    text: str,
    history_prompt: Optional[Dict[str,np.ndarray] | str],
    seconds: float,
    start_seconds: float,
    dna: SongDNA,
    segment: Segment,
    token_memory: PersistentTokenMemory,
    controls: Dict[str,float],
    cfg: KernelConfig,
) -> np.ndarray:
    """Direct Bark semantic loop with phase/harmony/token-prior bias."""
    if not BARK_AVAILABLE or torch is None:
        raise RuntimeError("Bark + torch are required for real generation")
    g=bark_generation
    missing=[k for k in BARK_CUSTOM_REQUIRED if not hasattr(g,k)]
    if missing:
        if cfg.strict_bark_compat:
            raise RuntimeError(f"Bark internals changed; custom sampler missing: {missing}")
        # Explicit fallback: generation still runs, but the music-aware semantic logit bias is disabled.
        return g.generate_text_semantic(
            text, history_prompt=history_prompt, temp=cfg.semantic_temp,
            top_k=cfg.semantic_top_k, top_p=cfg.semantic_top_p, silent=True,
            min_eos_p=None, max_gen_duration_s=seconds, allow_early_stop=False,
            use_kv_caching=True,
        )

    text=g._normalize_whitespace(text)
    if not text: text="[music]"
    if history_prompt is not None:
        hp=g._load_history_prompt(history_prompt)
        semantic_history=np.asarray(hp["semantic_prompt"],dtype=np.int64)[-256:]
        semantic_history=np.pad(semantic_history,(0,256-len(semantic_history)),constant_values=g.SEMANTIC_PAD_TOKEN)
    else:
        semantic_history=np.full(256,g.SEMANTIC_PAD_TOKEN,dtype=np.int64)

    if "text" not in g.models:
        g.preload_models()
    model_container=g.models["text"]
    model=model_container["model"]; tokenizer=model_container["tokenizer"]
    encoded=np.asarray(g._tokenize(tokenizer,text),dtype=np.int64)+g.TEXT_ENCODING_OFFSET
    encoded=encoded[:256]
    encoded=np.pad(encoded,(0,256-len(encoded)),constant_values=g.TEXT_PAD_TOKEN)

    if getattr(g,"OFFLOAD_CPU",False): model.to(g.models_devices["text"])
    device=next(model.parameters()).device
    x=torch.from_numpy(np.hstack([encoded,semantic_history,np.array([g.SEMANTIC_INFER_TOKEN])]).astype(np.int64))[None].to(device)
    semantic_rate=float(getattr(g,"SEMANTIC_RATE_HZ",49.9))
    max_steps=min(768,max(1,int(math.ceil(seconds*semantic_rate))))
    kv_cache=None
    generated=[]
    prev_token=int(semantic_history[-1]) if semantic_history[-1] < g.SEMANTIC_VOCAB_SIZE else None

    infer_ctx=getattr(g,"_inference_mode",None)
    context=infer_ctx() if infer_ctx is not None else torch.inference_mode()
    with context:
        for n in range(max_steps):
            x_input=x[:,[-1]] if kv_cache is not None else x
            logits,kv_cache=model(x_input,merge_context=True,use_cache=True,past_kv=kv_cache)
            scores=logits[0,0,:g.SEMANTIC_VOCAB_SIZE].float()

            absolute_t=start_seconds+n/semantic_rate
            beat_float=absolute_t/dna.seconds_per_beat
            beat_phase=beat_float%1.0
            phase_bin=int(beat_phase*cfg.phase_bins)%cfg.phase_bins
            bar_index=int(math.floor((absolute_t+1e-9)/dna.seconds_per_bar))
            harmonic_slot=dna.harmonic_slot_for_bar(bar_index)%cfg.harmonic_slots

            bias=token_memory.logit_bias(prev_token,phase_bin,harmonic_slot,controls,cfg)
            scores=scores+torch.from_numpy(bias).to(device=device,dtype=scores.dtype)

            # Downbeats become more deterministic; offbeats have slightly more room to mutate.
            downbeat_distance=min(beat_phase,1.0-beat_phase)
            lock=np.clip(1.0-downbeat_distance*4.0,0.0,1.0)*controls.get("beat_lock",0.5)
            local_temp=float(np.clip(cfg.semantic_temp*(1.08-0.20*lock)+0.10*segment.mutation*controls.get("exploration",0.5),0.34,0.88))
            tok=_sample_logits(scores,local_temp,cfg.semantic_top_k,cfg.semantic_top_p)
            generated.append(tok); prev_token=tok
            x=torch.cat((x,torch.tensor([[tok]],device=device,dtype=x.dtype)),dim=1)

    if getattr(g,"OFFLOAD_CPU",False): model.to("cpu")
    if hasattr(g,"_clear_cuda_cache"): g._clear_cuda_cache()
    return np.asarray(generated,dtype=np.int64)


class BarkMusicBackend:
    def __init__(self,cfg: KernelConfig):
        self.cfg=cfg; self.loaded=False
        self.sample_rate=24000
        self.semantic_rate=49.9

    def load(self):
        if self.loaded: return
        if not BARK_AVAILABLE:
            raise RuntimeError("Install Suno Bark first: pip install git+https://github.com/suno-ai/bark.git")
        report=bark_compat_report()
        if self.cfg.strict_bark_compat and not report["custom_sampler_compatible"]:
            raise RuntimeError(f"Bark custom-sampler compatibility failed: {report}")
        if self.cfg.allow_legacy_checkpoint_pickle:
            print("Loading official Bark checkpoints with scoped weights_only=False compatibility.")
        with legacy_bark_checkpoint_loading(self.cfg.allow_legacy_checkpoint_pickle):
            bark.preload_models()
        self.sample_rate=int(getattr(bark,"SAMPLE_RATE",24000))
        self.semantic_rate=float(getattr(bark_generation,"SEMANTIC_RATE_HZ",49.9))
        self.loaded=True

    def generate(self,prompt: str,history,seconds: float,start_seconds: float,dna: SongDNA,segment: Segment,token_memory: PersistentTokenMemory,controls: Dict[str,float],seed: int):
        self.load(); seed_everything(seed)
        g=bark_generation
        sem=music_generate_text_semantic(prompt,history,seconds,start_seconds,dna,segment,token_memory,controls,self.cfg)
        coarse_temp=float(np.clip(self.cfg.coarse_temp+0.08*segment.mutation*controls.get("exploration",0.5),0.38,0.82))
        coarse=g.generate_coarse(sem,history_prompt=history,temp=coarse_temp,top_k=self.cfg.coarse_top_k,top_p=self.cfg.coarse_top_p,silent=True,use_kv_caching=True)
        fine=g.generate_fine(coarse,history_prompt=history,temp=self.cfg.fine_temp,silent=True)
        audio=np.asarray(g.codec_decode(fine),dtype=np.float32).reshape(-1)
        hist={"semantic_prompt":np.asarray(sem),"coarse_prompt":np.asarray(coarse),"fine_prompt":np.asarray(fine)}
        return hist,peak_normalize(audio)


def compile_prompt(dna: SongDNA, seg: Segment) -> str:
    root_pc,quality,target_chroma,chord_name=dna.chord_for_bar(seg.bar_start)
    energy="sparse" if seg.energy<0.3 else "restrained" if seg.energy<0.5 else "driving" if seg.energy<0.75 else "intense"
    mutation="preserve motif" if seg.mutation<0.15 else "small evolution" if seg.mutation<0.35 else "controlled mutation" if seg.mutation<0.65 else "radical variation with same sonic identity"
    boundary=[]
    if seg.is_section_start: boundary.append("clear section entrance")
    if seg.is_section_end: boundary.append("coherent section handoff")
    extra="; ".join(boundary)
    if dna.vocal_mode=="singing" and seg.lyrics:
        lyric=" ".join(seg.lyrics.split())
        return (f"{dna.core_prompt()}; section {seg.section_name}; harmonic center {chord_name}; {energy}; {mutation}; {seg.direction}; {extra}; ♪ {lyric} ♪")[:1150]
    return (f"{dna.core_prompt()}; section {seg.section_name}; harmonic center {chord_name}; {energy}; {mutation}; {seg.direction}; {extra}; [music]")[:1150]


def motif_state(history: Dict[str,np.ndarray],audio: np.ndarray,sr: int,dna: SongDNA,token_memory: PersistentTokenMemory) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    sem=np.asarray(history["semantic_prompt"])
    fp=token_memory.fingerprint(sem,32)
    av=audio_state_vector(audio,sr,dna.bpm)
    chroma=chroma_vector(audio,sr)
    state=np.concatenate([av,fp])
    state=state/(np.linalg.norm(state)+1e-12)
    return state,chroma,fp


def target_controls(memory: RollingMusicMemory, token_memory: PersistentTokenMemory, target_chroma: np.ndarray, seg: Segment) -> Tuple[Dict[str,float],np.ndarray]:
    kin=memory.worldline.kinematics()
    drift=np.clip(kin["velocity"]*2.0+kin["q_js"],0,1)
    stagnation=np.clip(kin["stagnation"],0,1)
    if memory.macro_chroma is None: harmony_error=0.5
    else: harmony_error=np.clip(1.0-max(0,cosine(memory.macro_chroma,target_chroma)),0,1)
    pulse_error=0.35 if not memory.meso else np.clip(1.0-pulse_strength_from_memory(memory),0,1)
    token_entropy=np.clip(token_memory.token_entropy(),0,1)
    features=[drift,stagnation,harmony_error,pulse_error,token_entropy]
    q=QCTRL.run(features)
    # Stagnation should release exploration; hard drift should strengthen identity/memory.
    controls={
        "exploration":float(np.clip(0.25+0.55*q["exploration"]+0.25*stagnation-0.20*drift,0,1)),
        "motif_pull":float(np.clip(0.25+0.60*q["motif_pull"]+0.25*drift-0.15*stagnation,0,1)),
        "beat_lock":float(np.clip(0.30+0.60*q["beat_lock"],0,1)),
        "harmony_lock":float(np.clip(0.30+0.60*q["harmony_lock"]+0.20*harmony_error,0,1)),
        "memory_mix":float(np.clip(0.25+0.65*q["memory_mix"]+0.20*drift,0,1)),
    }
    return controls,q["probabilities"]


def pulse_strength_from_memory(memory: RollingMusicMemory) -> float:
    # Audio state stores pulse as its last component before normalization; use a conservative proxy.
    if not memory.meso: return 0.5
    vals=[]
    for m in list(memory.meso)[-4:]:
        vals.append(float(abs(m.audio_state[min(len(m.audio_state)-1,37)])))
    return float(np.clip(np.mean(vals)*3.0,0,1)) if vals else 0.5


class OptionalCLAPScorer:
    def __init__(self,enabled: bool=False,device: Optional[str]=None):
        self.enabled=enabled; self.device=device; self.model=None; self.processor=None

    def _load(self):
        if not self.enabled or self.model is not None: return
        try:
            from transformers import ClapModel, ClapProcessor
            self.processor=ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model=ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.device=self.device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()
        except Exception as e:
            print("CLAP unavailable; continuing without it:",e); self.enabled=False

    def score_instrumental(self,audio: np.ndarray,sr:int) -> float:
        if not self.enabled: return 0.0
        self._load()
        if not self.enabled: return 0.0
        n=max(1,int(len(audio)*48000/sr))
        src=np.linspace(0,1,len(audio),endpoint=False); dst=np.linspace(0,1,n,endpoint=False)
        y=np.interp(dst,src,audio).astype(np.float32)
        prompts=["instrumental music without vocals","spoken voice or singing"]
        inp=self.processor(text=prompts,audios=[y,y],sampling_rate=48000,return_tensors="pt",padding=True)
        inp={k:v.to(self.device) if hasattr(v,"to") else v for k,v in inp.items()}
        with torch.no_grad():
            out=self.model(**inp)
            te=out.text_embeds; ae=out.audio_embeds
            te=te/te.norm(dim=-1,keepdim=True); ae=ae/ae.norm(dim=-1,keepdim=True)
            sims=(ae[0:1]@te.T).squeeze(0)
            return float((sims[0]-sims[1]).cpu())


def candidate_score(
    previous_audio: Optional[np.ndarray], audio: np.ndarray, sr: int, dna: SongDNA,
    target_chroma: np.ndarray, memory: RollingMusicMemory, state: np.ndarray,
    token_fp: np.ndarray, controls: Dict[str,float], cfg: KernelConfig,
    clap_scorer: Optional[OptionalCLAPScorer]=None,
) -> Tuple[float,Dict[str,float]]:
    boundary=boundary_similarity(previous_audio,audio,sr)
    timbre=0.5 if memory.macro_state is None else (cosine(state,memory.macro_state)+1)/2
    pulse=pulse_strength(audio,sr,dna.bpm)
    ch=chroma_vector(audio,sr)
    harmony=(cosine(ch,target_chroma)+1)/2
    motif=0.5 if memory.macro_token_fp is None else (cosine(token_fp,memory.macro_token_fp)+1)/2
    novelty=1.0-timbre
    target_novelty=0.18+0.55*controls.get("exploration",0.5)
    novelty_fit=1.0-min(1.0,abs(novelty-target_novelty)/0.65)
    level_fit=1.0-min(1.0,abs(rms(audio)-0.12)/0.20)
    score=(cfg.w_boundary*boundary+cfg.w_timbre*timbre+cfg.w_pulse*pulse+cfg.w_harmony*harmony+cfg.w_motif*motif+cfg.w_novelty*novelty_fit+cfg.w_level*level_fit)
    clap=0.0
    if clap_scorer is not None and cfg.instrumental_only:
        clap=clap_scorer.score_instrumental(audio,sr)
        score += 0.4*clap
    parts={"boundary":boundary,"timbre":timbre,"pulse":pulse,"harmony":harmony,"motif":motif,"novelty_fit":novelty_fit,"level_fit":level_fit,"clap":clap}
    return float(score),parts


@dataclass
class RenderedSegment:
    segment: Segment
    prompt: str
    audio: np.ndarray
    history: Dict[str,np.ndarray]
    score: float
    score_parts: Dict[str,float]
    controls: Dict[str,float]
    q_probs: np.ndarray
    candidate_index: int


class BarkQuantumMusicEngine:
    def __init__(self,cfg: Optional[KernelConfig]=None,backend: Optional[BarkMusicBackend]=None):
        self.cfg=cfg or KernelConfig()
        self.backend=backend or BarkMusicBackend(self.cfg)
        self.token_memory=PersistentTokenMemory(10_000,self.cfg.phase_bins,self.cfg.harmonic_slots,self.cfg.max_bigram_followers)
        self.memory=RollingMusicMemory(self.cfg,self.token_memory)
        self.clap=OptionalCLAPScorer(self.cfg.use_clap)

    def _harmonic_slot_fn(self,dna: SongDNA):
        def _slot(seconds: float) -> int:
            bar=int(math.floor((float(seconds)+1e-9)/dna.seconds_per_bar))
            return dna.harmonic_slot_for_bar(bar)%self.cfg.harmonic_slots
        return _slot

    def _conditioning(self,target_chroma: np.ndarray,dna: SongDNA):
        # First vocal segment can start from an upstream Bark voice preset; after that,
        # accepted generated history becomes the singer/timbre identity.
        if self.memory.recent_history is None and dna.voice_preset:
            return dna.voice_preset
        exemplar=self.memory.retrieve_exemplar(target_chroma,self.memory.macro_state)
        epi=exemplar.history if exemplar is not None else None
        return fuse_many_histories([
            (self.memory.anchor_history,self.cfg.macro_anchor_tokens),
            (epi,self.cfg.episodic_tokens),
            (self.memory.recent_history,self.cfg.recent_tokens),
        ])

    def render_stream(self,plan: SongPlan) -> Generator[RenderedSegment,None,None]:
        self.backend.load()
        sr=self.backend.sample_rate
        outdir=Path(self.cfg.output_dir); (outdir/"chunks").mkdir(parents=True,exist_ok=True); (outdir/"state").mkdir(parents=True,exist_ok=True)
        previous_audio=None
        manifest={"created":time.time(),"dna":asdict(plan.dna),"config":asdict(self.cfg),"segments":[]}

        for seg in plan.segments():
            # The control clock follows the score/arrangement, not Bark's slightly variable output duration.
            planned_start_seconds=float(seg.bar_start*plan.dna.seconds_per_bar)
            _,_,target_chroma,chord_name=plan.dna.chord_for_bar(seg.bar_start)
            controls,q_probs=target_controls(self.memory,self.token_memory,target_chroma,seg)
            prompt=compile_prompt(plan.dna,seg)
            conditioning=self._conditioning(target_chroma,plan.dna)
            best=None
            for ci in range(max(1,self.cfg.candidates_per_segment)):
                seed=self.cfg.seed+seg.index*1009+ci*97
                history,audio=self.backend.generate(prompt,conditioning,seg.seconds,planned_start_seconds,plan.dna,seg,self.token_memory,controls,seed)
                state,chroma,fp=motif_state(history,audio,sr,plan.dna,self.token_memory)
                score,parts=candidate_score(previous_audio,audio,sr,plan.dna,target_chroma,self.memory,state,fp,controls,self.cfg,self.clap)
                candidate=RenderedSegment(seg,prompt,audio,history,score,parts,controls,q_probs,ci)
                if best is None or score>best.score: best=candidate
            assert best is not None

            # Commit ONLY the accepted candidate. Rejected candidates cannot poison long-term state.
            state,chroma,fp=motif_state(best.history,best.audio,sr,plan.dna,self.token_memory)
            mem=SegmentMemory(seg.index,seg.bar_start,planned_start_seconds,best.history,state,chroma,fp,best.score,best.q_probs,plan.dna.harmonic_slot_for_bar(seg.bar_start)%self.cfg.harmonic_slots)
            self.memory.commit(mem,self.backend.semantic_rate,plan.dna.bpm,self._harmonic_slot_fn(plan.dna))
            # Fill the last worldline point with actual candidate metrics.
            if self.memory.worldline.points:
                p=self.memory.worldline.points[-1]
                p.harmony_score=best.score_parts["harmony"]; p.pulse_score=best.score_parts["pulse"]; p.motif_score=best.score_parts["motif"]

            if self.cfg.keep_chunk_wavs:
                write_wav(outdir/"chunks"/f"{seg.index:03d}_{seg.section_name}.wav",sr,best.audio)
            if self.cfg.keep_history_npz:
                np.savez_compressed(outdir/"state"/f"{seg.index:03d}.npz",**best.history)

            manifest["segments"].append({
                "index":seg.index,"section":seg.section_name,"bar_start":seg.bar_start,"bars":seg.bars,
                "target_seconds":seg.seconds,"planned_start_seconds":planned_start_seconds,
                "rendered_seconds":len(best.audio)/sr,"candidate":best.candidate_index,"score":best.score,
                "score_parts":best.score_parts,"controls":best.controls,"prompt":best.prompt,
                "worldline":self.memory.worldline.kinematics(),"chord":chord_name,
            })
            (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
            previous_audio=best.audio
            yield best

    def render(self,plan: SongPlan,output_wav: str="bark_quantum_song.wav") -> np.ndarray:
        full=np.zeros(0,dtype=np.float32); sr=None
        for seg in self.render_stream(plan):
            sr=self.backend.sample_rate
            full=seg.audio if full.size==0 else equal_power_join(full,seg.audio,sr,self.cfg.crossfade_ms)
            print(f"[{seg.segment.index:03d}] {seg.segment.section_name:<12} score={seg.score:.3f} candidate={seg.candidate_index} controls={{{', '.join(f'{k}:{v:.2f}' for k,v in seg.controls.items())}}}")
        if sr is None: raise RuntimeError("No segments rendered")
        full=peak_normalize(full)
        write_wav(output_wav,sr,full)
        print("Saved",output_wav,"seconds=",len(full)/sr)
        return full


def legacy_elements_to_plan(song_elements: Sequence[Dict[str,Any]], dna: Optional[SongDNA]=None) -> SongPlan:
    dna=dna or SongDNA(title="Legacy Rewired",bpm=120.0,vocal_mode="singing")
    sections=[]
    for i,e in enumerate(song_elements):
        content=e.get("lyrics") or e.get("music") or e.get("end")
        if not content: continue
        is_lyrics="lyrics" in e
        pause=float(e.get("pause",e.get("[music]pause",0.0)) or 0.0)
        # Convert each legacy element to at least one musical bar. Pause becomes arrangement space, not an unrelated silence task.
        approx_bars=max(1,int(round((2.0+pause)/dna.seconds_per_bar)))
        sections.append(Section(
            name=f"legacy_{i:02d}", bars=approx_bars,
            direction=str(content), energy=0.45 if i<2 else 0.6,
            mutation=0.12 if i>0 else 0.25,
            lyrics=str(content).replace("[music]","").strip() if is_lyrics else "",
        ))
    return SongPlan(dna,sections,segment_bars=1,max_segment_seconds=CFG.max_segment_seconds)


def apply_runtime_profile(cfg: KernelConfig, name: str="balanced") -> KernelConfig:
    name=name.lower().strip()
    if name=="draft":
        cfg.candidates_per_segment=1
        cfg.use_clap=False
        cfg.keep_history_npz=False
    elif name=="balanced":
        cfg.candidates_per_segment=2
        cfg.use_clap=False
        cfg.keep_history_npz=True
    elif name=="quality":
        cfg.candidates_per_segment=4
        cfg.use_clap=True
        cfg.keep_history_npz=True
    else:
        raise ValueError("profile must be: draft, balanced, or quality")
    return cfg

# Example:
# apply_runtime_profile(CFG, "balanced")


def twitch_spool(plan: SongPlan, cfg: Optional[KernelConfig]=None, spool_dir: str="twitch_spool"):
    cfg=cfg or KernelConfig()
    cfg.output_dir=spool_dir
    engine=BarkQuantumMusicEngine(cfg)
    index_file=Path(spool_dir)/"now_playing.json"
    for rendered in engine.render_stream(plan):
        payload={
            "segment":rendered.segment.index,
            "section":rendered.segment.section_name,
            "score":rendered.score,
            "chunk":str(Path(spool_dir)/"chunks"/f"{rendered.segment.index:03d}_{rendered.segment.section_name}.wav"),
            "controls":rendered.controls,
        }
        tmp=index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload,indent=2)); tmp.replace(index_file)
        yield payload

# Example live loop:
# for event in twitch_spool(plan, CFG):
#     print(event)


def plot_worldline(engine: BarkQuantumMusicEngine):
    import matplotlib.pyplot as plt
    pts=list(engine.memory.worldline.points)
    if not pts:
        print("No worldline points yet"); return
    x=np.arange(len(pts))
    hs=[p.harmony_score for p in pts]; ps=[p.pulse_score for p in pts]; ms=[p.motif_score for p in pts]
    qh=[entropy_bits(p.quantum_probs) for p in pts]
    plt.figure(figsize=(12,4))
    plt.plot(x,hs,label="harmony")
    plt.plot(x,ps,label="pulse")
    plt.plot(x,ms,label="motif")
    plt.plot(x,np.asarray(qh)/max(qh+[1e-9]),label="quantum-state entropy / max")
    plt.xlabel("accepted segment"); plt.ylim(0,1.05); plt.grid(alpha=.2); plt.legend(); plt.show()


def save_engine_checkpoint(engine: BarkQuantumMusicEngine,path: str="music_memory_checkpoint.npz"):
    """Persist learned token priors + macro identity + Bark anchor/recent histories."""
    mem=engine.memory; tm=engine.token_memory
    bigram_json=json.dumps({str(k):dict(v) for k,v in tm.bigram.items()})
    payload={
        "unigram":tm.unigram,
        "phase_counts":tm.phase_counts,
        "harmony_counts":tm.harmony_counts,
        "token_total":np.array([tm.total],dtype=np.int64),
        "bigram_json":np.array([bigram_json]),
        "macro_state":np.array([]) if mem.macro_state is None else mem.macro_state,
        "macro_chroma":np.array([]) if mem.macro_chroma is None else mem.macro_chroma,
        "macro_token_fp":np.array([]) if mem.macro_token_fp is None else mem.macro_token_fp,
    }
    for prefix,h in (("anchor",mem.anchor_history),("recent",mem.recent_history)):
        if h is not None:
            payload[f"{prefix}_semantic"]=h["semantic_prompt"]
            payload[f"{prefix}_coarse"]=h["coarse_prompt"]
            payload[f"{prefix}_fine"]=h["fine_prompt"]
    np.savez_compressed(path,**payload)
    print("Saved",path)


def load_engine_checkpoint(engine: BarkQuantumMusicEngine,path: str="music_memory_checkpoint.npz"):
    z=np.load(path,allow_pickle=False)
    tm=engine.token_memory; mem=engine.memory
    tm.unigram=np.asarray(z["unigram"],dtype=np.float64)
    tm.phase_counts=np.asarray(z["phase_counts"],dtype=np.float32)
    tm.harmony_counts=np.asarray(z["harmony_counts"],dtype=np.float32)
    tm.total=int(np.asarray(z["token_total"]).reshape(-1)[0])
    raw=json.loads(str(np.asarray(z["bigram_json"]).reshape(-1)[0]))
    tm.bigram=defaultdict(Counter,{int(k):Counter({int(t):int(n) for t,n in v.items()}) for k,v in raw.items()})
    for name in ("macro_state","macro_chroma","macro_token_fp"):
        arr=np.asarray(z[name]); setattr(mem,name,None if arr.size==0 else arr)
    for prefix in ("anchor","recent"):
        keys=[f"{prefix}_semantic",f"{prefix}_coarse",f"{prefix}_fine"]
        if all(k in z.files for k in keys):
            h={"semantic_prompt":np.asarray(z[keys[0]]),"coarse_prompt":np.asarray(z[keys[1]]),"fine_prompt":np.asarray(z[keys[2]])}
            setattr(mem,f"{prefix}_history",h)
    print("Loaded",path,"tokens=",tm.total)

# ============================================================================
# v3 — SCORE-BEFORE-SOUND HYBRID COMPOSER
# ----------------------------------------------------------------------------
# Bark is excellent at neural audio texture, but it does not expose explicit
# note/rest/voice control. v3 therefore composes a symbolic score first and
# renders a deterministic musical core. Bark can then be mixed as a low-level
# texture stem instead of being asked to invent the composition itself.
# ============================================================================

from dataclasses import replace

@dataclass(order=True)
class NoteEvent:
    start_beat: float
    duration_beats: float
    midi: int
    velocity: float
    part: str = "lead"
    pan: float = 0.0


@dataclass(order=True)
class DrumHit:
    start_beat: float
    kind: str
    velocity: float


@dataclass
class SymbolicScore:
    bpm: float
    beats_per_bar: int
    total_bars: int
    notes: List[NoteEvent] = field(default_factory=list)
    drums: List[DrumHit] = field(default_factory=list)
    section_boundaries: List[Tuple[int, str]] = field(default_factory=list)
    motif_degrees: Tuple[int, ...] = ()
    motif_rhythm: Tuple[float, ...] = ()

    @property
    def total_beats(self) -> float:
        return float(self.total_bars * self.beats_per_bar)

    @property
    def seconds(self) -> float:
        return self.total_beats * 60.0 / self.bpm

    def notes_for_part(self, part: str) -> List[NoteEvent]:
        return [n for n in self.notes if n.part == part]


@dataclass
class ComposerConfig:
    lead_octave: int = 5
    bass_octave: int = 2
    chord_octave: int = 3
    motif_bars: int = 2
    motif_repeat_every_bars: int = 4
    swing: float = 0.03
    humanize_ms: float = 5.0
    use_drums: bool = True
    use_bass: bool = True
    use_harmony: bool = True
    use_lead: bool = True
    cadence_strength: float = 0.85


def midi_from_pc(pc: int, octave: int) -> int:
    # MIDI C-1 = 0, therefore C4 = 60.
    return int(12 * (int(octave) + 1) + (int(pc) % 12))


def chord_intervals(quality: str) -> Tuple[int, ...]:
    return {
        "major": (0, 4, 7), "minor": (0, 3, 7), "dim": (0, 3, 6),
        "sus2": (0, 2, 7), "sus4": (0, 5, 7), "maj7": (0, 4, 7, 11),
        "min7": (0, 3, 7, 10), "dom7": (0, 4, 7, 10),
    }.get(str(quality).lower(), (0, 3, 7))


def _nearest_scale_midi(pc: int, target_midi: int) -> int:
    octave = target_midi // 12 - 1
    candidates = [midi_from_pc(pc, octave + k) for k in (-1, 0, 1, 2)]
    return int(min(candidates, key=lambda m: abs(m - target_midi)))


class SymbolicComposer:
    """Phrase-level deterministic composer.

    The score is explicit: notes, durations, parts, and drum hits. Repetition is
    deliberate and variation is constrained. This is the layer that v2 lacked.
    """

    RHYTHM_BANK = (
        (0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0),
        (1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 2.0),
        (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 2.0, 1.0),
    )
    DEGREE_BANK = (
        (0, 2, 3, 4, 3, 2, 0, 6, 0),
        (0, 1, 2, 4, 3, 2, 1, 6, 0),
        (0, 2, 4, 3, 2, 1, 2, 6, 0, 0),
    )

    def __init__(self, cfg: Optional[ComposerConfig] = None, seed: int = 1337):
        self.cfg = cfg or ComposerConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _new_motif(self, dna: SongDNA) -> Tuple[List[int], List[float]]:
        i = int(self.rng.integers(0, min(len(self.RHYTHM_BANK), len(self.DEGREE_BANK))))
        rhythm = list(self.RHYTHM_BANK[i])
        degrees = list(self.DEGREE_BANK[i])
        target = self.cfg.motif_bars * dna.meter_numerator
        # Normalize the cell to exactly the motif span.
        s = sum(rhythm)
        rhythm = [r * target / s for r in rhythm]
        return degrees, rhythm

    @staticmethod
    def _section_for_bar(plan: SongPlan, bar: int) -> Tuple[Section, int]:
        cursor = 0
        for sec in plan.sections:
            if cursor <= bar < cursor + sec.bars:
                return sec, bar - cursor
            cursor += sec.bars
        return plan.sections[-1], max(0, bar - cursor)

    def _mutated_motif(self, base_degrees: Sequence[int], mutation: float, phrase_index: int) -> List[int]:
        d = list(base_degrees)
        if phrase_index == 0 or mutation < 0.08:
            return d
        local = np.random.default_rng(self.seed + phrase_index * 7919)
        # Variation is musical: neighbor motion, octave implication, or inversion;
        # not arbitrary token noise.
        max_changes = max(1, int(round(mutation * len(d) * 0.45)))
        for idx in local.choice(len(d), size=min(max_changes, len(d)), replace=False):
            if idx in (0, len(d) - 1):
                continue
            d[idx] = int(np.clip(d[idx] + local.choice([-1, 1]), 0, 6))
        if mutation > 0.62 and phrase_index % 3 == 2:
            middle = d[1:-1]
            d[1:-1] = middle[::-1]
        return d

    def _compose_lead_bar(self, score: SymbolicScore, dna: SongDNA, bar: int,
                          motif_degrees: Sequence[int], motif_rhythm: Sequence[float],
                          sec: Section, local_bar: int) -> None:
        bpb = dna.meter_numerator
        motif_len = self.cfg.motif_bars * bpb
        phrase = bar // self.cfg.motif_bars
        degrees = self._mutated_motif(motif_degrees, sec.mutation, phrase)
        motif_start = (bar // self.cfg.motif_bars) * motif_len
        bar_start = bar * bpb
        bar_end = bar_start + bpb
        scale = dna.scale_pcs()
        root_midi = midi_from_pc(dna.root_pc, self.cfg.lead_octave)

        t = motif_start
        note_index = 0
        for dur in motif_rhythm:
            note_start = t
            note_end = t + dur
            t = note_end
            if note_end <= bar_start + 1e-9:
                note_index += 1
                continue
            if note_start >= bar_end - 1e-9:
                break
            degree = degrees[note_index % len(degrees)] % len(scale)
            pc = scale[degree]
            # Track near a stable tessitura and gently react to the local chord.
            chord_root_pc, _, _, _ = dna.chord_for_bar(bar)
            target = root_midi + (degree - 2)
            midi = _nearest_scale_midi(pc, target)
            if degree in (0, 2, 4) and ((pc - chord_root_pc) % 12) not in chord_intervals(dna.chord_for_bar(bar)[1]):
                midi = _nearest_scale_midi(chord_root_pc, midi)

            clipped_start = max(note_start, bar_start)
            clipped_end = min(note_end, bar_end)
            dur2 = clipped_end - clipped_start
            if dur2 > 0.08:
                # Stronger phrase openings; ending bars cadence toward tonic.
                accent = 0.10 if abs((clipped_start - bar_start) % bpb) < 1e-6 else 0.0
                vel = float(np.clip(0.52 + 0.32 * sec.energy + accent, 0.25, 0.98))
                if local_bar == sec.bars - 1 and clipped_end >= bar_end - 0.1 and self.rng.random() < self.cfg.cadence_strength:
                    midi = _nearest_scale_midi(dna.root_pc, midi)
                score.notes.append(NoteEvent(clipped_start, max(0.10, dur2 * 0.92), midi, vel, "lead", 0.12))
            note_index += 1

    def _compose_harmony_bar(self, score: SymbolicScore, dna: SongDNA, bar: int, sec: Section) -> None:
        bpb = dna.meter_numerator
        root_pc, quality, _, _ = dna.chord_for_bar(bar)
        tones = chord_intervals(quality)
        root = midi_from_pc(root_pc, self.cfg.chord_octave)
        notes = [root + iv for iv in tones]
        start = bar * bpb
        vel = float(np.clip(0.28 + 0.28 * sec.energy, 0.22, 0.68))
        if sec.energy < 0.58:
            for n in notes:
                score.notes.append(NoteEvent(start, bpb * 0.92, n, vel, "pad", -0.10 if n == notes[0] else 0.10))
        else:
            # Arpeggio turns static chord tones into meter-aware motion.
            step = 0.5
            count = int(round(bpb / step))
            order = list(range(len(notes))) + list(range(len(notes) - 2, 0, -1))
            for i in range(count):
                n = notes[order[i % len(order)]]
                score.notes.append(NoteEvent(start + i * step, step * 0.82, n + (12 if sec.energy > 0.82 and i % 4 == 3 else 0), vel * 0.88, "arp", 0.18))

    def _compose_bass_bar(self, score: SymbolicScore, dna: SongDNA, bar: int, sec: Section) -> None:
        bpb = dna.meter_numerator
        root_pc, _, _, _ = dna.chord_for_bar(bar)
        root = midi_from_pc(root_pc, self.cfg.bass_octave)
        start = bar * bpb
        vel = float(np.clip(0.46 + 0.34 * sec.energy, 0.35, 0.92))
        pattern = [0, 0, 7, 0]
        for beat in range(bpb):
            if sec.energy < 0.35 and beat % 2:
                continue
            offset = pattern[beat % len(pattern)]
            dur = 0.72 if sec.energy > 0.55 else 0.88
            score.notes.append(NoteEvent(start + beat, dur, root + offset, vel * (1.0 if beat == 0 else 0.84), "bass", -0.05))

    def _compose_drums_bar(self, score: SymbolicScore, dna: SongDNA, bar: int, sec: Section) -> None:
        bpb = dna.meter_numerator
        start = bar * bpb
        e = sec.energy
        for beat in range(bpb):
            # Kick anchors 1 and 3; higher energy adds syncopation.
            if beat in (0, 2) or (e > 0.78 and beat == 3):
                score.drums.append(DrumHit(start + beat, "kick", float(np.clip(0.62 + 0.30 * e, 0, 1))))
            if beat in (1, 3):
                score.drums.append(DrumHit(start + beat, "snare", float(np.clip(0.48 + 0.32 * e, 0, 1))))
            # Eighth hats make the pulse audible as rhythm rather than a held tone.
            score.drums.append(DrumHit(start + beat, "hat", 0.28 + 0.30 * e))
            if e > 0.28:
                score.drums.append(DrumHit(start + beat + 0.5, "hat", 0.22 + 0.24 * e))
        if e > 0.68:
            score.drums.append(DrumHit(start + 2.5, "kick", 0.48 + 0.20 * e))
        if bar % 4 == 3 and e > 0.52:
            for x in (3.0, 3.25, 3.5, 3.75):
                score.drums.append(DrumHit(start + x, "hat_open" if x == 3.75 else "hat", 0.44 + 0.20 * e))

    def compose(self, plan: SongPlan) -> SymbolicScore:
        dna = plan.dna
        if dna.meter_denominator != 4:
            raise ValueError("v3 symbolic composer currently expects a quarter-note denominator (e.g. 4/4 or 3/4).")
        total_bars = sum(s.bars for s in plan.sections)
        score = SymbolicScore(dna.bpm, dna.meter_numerator, total_bars)
        motif_degrees, motif_rhythm = self._new_motif(dna)
        score.motif_degrees = tuple(motif_degrees)
        score.motif_rhythm = tuple(motif_rhythm)

        cursor = 0
        for sec in plan.sections:
            score.section_boundaries.append((cursor, sec.name))
            for local_bar in range(sec.bars):
                bar = cursor + local_bar
                if self.cfg.use_lead:
                    self._compose_lead_bar(score, dna, bar, motif_degrees, motif_rhythm, sec, local_bar)
                if self.cfg.use_harmony:
                    self._compose_harmony_bar(score, dna, bar, sec)
                if self.cfg.use_bass:
                    self._compose_bass_bar(score, dna, bar, sec)
                if self.cfg.use_drums:
                    self._compose_drums_bar(score, dna, bar, sec)
            cursor += sec.bars

        score.notes.sort()
        score.drums.sort()
        return score


@dataclass
class SynthConfig:
    sample_rate: int = 44100
    master_gain: float = 0.82
    reverb_mix: float = 0.12
    soft_clip_drive: float = 1.25


def _midi_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = min(n, int(max(1, attack * sr)))
    d = min(n - a, int(max(1, decay * sr))) if n > a else 0
    r = min(max(1, int(release * sr)), max(1, n - a - d))
    s = max(0, n - a - d - r)
    parts = []
    if a:
        parts.append(np.linspace(0.0, 1.0, a, endpoint=False))
    if d:
        parts.append(np.linspace(1.0, sustain, d, endpoint=False))
    if s:
        parts.append(np.full(s, sustain))
    if r:
        parts.append(np.linspace(sustain, 0.0, r, endpoint=True))
    env = np.concatenate(parts) if parts else np.zeros(n)
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n].astype(np.float32)


def _oscillator(freq: float, seconds: float, sr: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(round(seconds * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    phase = rng.random() * 2 * np.pi
    if kind == "bass":
        x = 0.76 * np.sin(2*np.pi*freq*t + phase) + 0.20 * np.sin(2*np.pi*2*freq*t + 0.4*phase)
    elif kind in ("pad", "arp"):
        # Band-limited-ish additive waveform; no naive saw alias storm.
        x = np.zeros_like(t)
        harmonics = 6 if kind == "arp" else 4
        for h in range(1, harmonics + 1):
            x += (1.0 / (h ** 1.35)) * np.sin(2*np.pi*freq*h*t + phase*h)
        x /= max(1.0, np.max(np.abs(x)))
    else:  # lead
        vibrato = 0.0022 * np.sin(2*np.pi*5.1*t)
        x = 0.68*np.sin(2*np.pi*freq*(1+vibrato)*t + phase)
        x += 0.22*np.sin(2*np.pi*2*freq*t + phase*1.7)
        x += 0.10*np.sin(2*np.pi*3*freq*t + phase*0.7)
    return x.astype(np.float32)


def _stereo_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1).astype(np.float32)


class ScoreSynthesizer:
    def __init__(self, cfg: Optional[SynthConfig] = None, seed: int = 1337):
        self.cfg = cfg or SynthConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _note_audio(self, ev: NoteEvent, bpm: float) -> np.ndarray:
        sr = self.cfg.sample_rate
        sec_per_beat = 60.0 / bpm
        dur_s = max(0.04, ev.duration_beats * sec_per_beat)
        tail = 0.18 if ev.part in ("pad",) else 0.08
        kind = ev.part if ev.part in ("bass", "pad", "arp") else "lead"
        x = _oscillator(_midi_hz(ev.midi), dur_s + tail, sr, kind, self.rng)
        if ev.part == "pad":
            env = _adsr(len(x), sr, 0.10, 0.22, 0.72, min(0.35, dur_s * 0.35))
            gain = 0.20
        elif ev.part == "bass":
            env = _adsr(len(x), sr, 0.008, 0.08, 0.78, 0.07)
            gain = 0.28
        elif ev.part == "arp":
            env = _adsr(len(x), sr, 0.004, 0.08, 0.36, 0.08)
            gain = 0.18
        else:
            env = _adsr(len(x), sr, 0.006, 0.12, 0.55, 0.08)
            gain = 0.24
        x = x * env * float(ev.velocity) * gain
        return _stereo_pan(x, ev.pan)

    def _drum_audio(self, hit: DrumHit) -> np.ndarray:
        sr = self.cfg.sample_rate
        rng = self.rng
        if hit.kind == "kick":
            dur = 0.36
            n = int(sr * dur)
            t = np.arange(n) / sr
            f = 130.0 * np.exp(-t * 16.0) + 44.0
            phase = 2*np.pi*np.cumsum(f)/sr
            body = np.sin(phase) * np.exp(-t*11.0)
            click = rng.standard_normal(n) * np.exp(-t*85.0) * 0.08
            mono = (body + click) * hit.velocity * 0.72
            return _stereo_pan(mono.astype(np.float32), 0.0)
        if hit.kind == "snare":
            dur = 0.24
            n = int(sr * dur)
            t = np.arange(n) / sr
            noise = rng.standard_normal(n)
            tone = np.sin(2*np.pi*185*t)
            mono = (0.68*noise*np.exp(-t*18.0) + 0.32*tone*np.exp(-t*13.0)) * hit.velocity * 0.28
            return _stereo_pan(mono.astype(np.float32), 0.05)
        dur = 0.12 if hit.kind == "hat" else 0.28
        n = int(sr * dur)
        t = np.arange(n) / sr
        noise = rng.standard_normal(n)
        # Crude differentiator suppresses lows and makes metallic hats.
        hp = np.concatenate([[0.0], np.diff(noise)])
        env = np.exp(-t * (34.0 if hit.kind == "hat" else 13.0))
        mono = hp * env * hit.velocity * 0.10
        return _stereo_pan(mono.astype(np.float32), 0.28)

    @staticmethod
    def _simple_reverb(x: np.ndarray, sr: int, mix: float) -> np.ndarray:
        if mix <= 0:
            return x
        wet = np.zeros_like(x)
        taps = ((0.071, 0.33), (0.113, 0.24), (0.173, 0.17), (0.229, 0.11))
        for delay, gain in taps:
            d = int(sr * delay)
            if d < len(x):
                wet[d:] += x[:-d] * gain
        return (x * (1.0 - mix) + wet * mix).astype(np.float32)

    def render(self, score: SymbolicScore, tail_seconds: float = 1.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        sr = self.cfg.sample_rate
        total = int(round((score.seconds + tail_seconds) * sr))
        parts = {p: np.zeros((total, 2), dtype=np.float32) for p in ("lead", "bass", "pad", "arp", "drums")}
        sec_per_beat = 60.0 / score.bpm

        for ev in score.notes:
            audio = self._note_audio(ev, score.bpm)
            start = int(round(ev.start_beat * sec_per_beat * sr))
            end = min(total, start + len(audio))
            if end > start:
                parts[ev.part][start:end] += audio[:end-start]

        for hit in score.drums:
            audio = self._drum_audio(hit)
            start = int(round(hit.start_beat * sec_per_beat * sr))
            end = min(total, start + len(audio))
            if end > start:
                parts["drums"][start:end] += audio[:end-start]

        # Gentle per-stem leveling.
        for k, x in parts.items():
            m = float(np.max(np.abs(x))) if x.size else 0.0
            if m > 1.0:
                parts[k] = x / m

        mix = sum(parts.values())
        mix = self._simple_reverb(mix, sr, self.cfg.reverb_mix)
        drive = max(1e-5, self.cfg.soft_clip_drive)
        mix = np.tanh(mix * drive) / np.tanh(drive)
        mix *= self.cfg.master_gain
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.985:
            mix *= 0.985 / peak
        return mix.astype(np.float32), parts


def write_stereo_wav(path: str | Path, sr: int, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.985:
        x = x * (0.985 / peak)
    scipy_write_wav(path, int(sr), np.clip(x * 32767.0, -32768, 32767).astype(np.int16))


def export_midi(score: SymbolicScore, path: str | Path, program_map: Optional[Dict[str, int]] = None) -> str:
    """Export the exact symbolic composition. Requires `mido` (small dependency)."""
    try:
        import mido
    except Exception as exc:
        raise RuntimeError("MIDI export requires: pip install mido") from exc

    program_map = program_map or {"lead": 81, "bass": 38, "pad": 89, "arp": 10}
    ticks_per_beat = 480
    mf = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(score.bpm)

    for channel, part in enumerate(("lead", "bass", "pad", "arp")):
        tr = mido.MidiTrack(); mf.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=part, time=0))
        if channel == 0:
            tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        tr.append(mido.Message("program_change", program=int(program_map.get(part, 0)), channel=channel, time=0))
        events = []
        for n in score.notes_for_part(part):
            st = int(round(n.start_beat * ticks_per_beat))
            en = int(round((n.start_beat + n.duration_beats) * ticks_per_beat))
            vel = int(np.clip(round(n.velocity * 127), 1, 127))
            events.append((st, 1, mido.Message("note_on", note=int(n.midi), velocity=vel, channel=channel, time=0)))
            events.append((en, 0, mido.Message("note_off", note=int(n.midi), velocity=0, channel=channel, time=0)))
        events.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tick, _, msg in events:
            msg.time = max(0, tick - last); tr.append(msg); last = tick

    # General MIDI drums use channel 10 => zero-indexed channel 9.
    drum_map = {"kick": 36, "snare": 38, "hat": 42, "hat_open": 46}
    tr = mido.MidiTrack(); mf.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="drums", time=0))
    events = []
    for h in score.drums:
        st = int(round(h.start_beat * ticks_per_beat))
        en = st + int(round(0.08 * ticks_per_beat))
        note = drum_map.get(h.kind, 42)
        vel = int(np.clip(round(h.velocity * 127), 1, 127))
        events.append((st, 1, mido.Message("note_on", note=note, velocity=vel, channel=9, time=0)))
        events.append((en, 0, mido.Message("note_off", note=note, velocity=0, channel=9, time=0)))
    events.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last); tr.append(msg); last = tick

    path = str(path)
    mf.save(path)
    return path


@dataclass
class HybridRenderResult:
    score: SymbolicScore
    audio: np.ndarray
    sample_rate: int
    stems: Dict[str, np.ndarray]
    bark_texture: Optional[np.ndarray] = None


class BarkScoreHybridEngine:
    """v3 engine: symbolic composition is primary; Bark is optional texture.

    This intentionally inverts v2's responsibility split. The composition exists
    before Bark is called, so the result remains music even if Bark is disabled.
    """

    def __init__(self, kernel_cfg: Optional[KernelConfig] = None,
                 composer_cfg: Optional[ComposerConfig] = None,
                 synth_cfg: Optional[SynthConfig] = None):
        self.kernel_cfg = kernel_cfg or KernelConfig()
        self.composer = SymbolicComposer(composer_cfg, self.kernel_cfg.seed)
        self.synth = ScoreSynthesizer(synth_cfg, self.kernel_cfg.seed)

    @staticmethod
    def _fit_texture(texture: np.ndarray, n: int) -> np.ndarray:
        x = np.asarray(texture, dtype=np.float32).reshape(-1)
        if len(x) == 0:
            return np.zeros((n, 2), dtype=np.float32)
        if len(x) < n:
            reps = int(np.ceil(n / len(x)))
            x = np.tile(x, reps)
        x = x[:n]
        # Bark is centered and kept deliberately quiet. It is texture, not score.
        return np.stack([x, x], axis=1)

    def render(self, plan: SongPlan, output_wav: str = "bark_score_hybrid.wav",
               midi_path: Optional[str] = "bark_score_hybrid.mid",
               use_bark_texture: bool = False, bark_texture_mix: float = 0.10) -> HybridRenderResult:
        score = self.composer.compose(plan)
        audio, stems = self.synth.render(score)
        bark_tex = None

        if use_bark_texture:
            if not BARK_AVAILABLE:
                print("Bark unavailable; rendering the musical score without neural texture.")
            else:
                # Ask the v2 Bark engine for texture only. Its musical responsibility
                # has been demoted: a bad Bark segment can no longer erase the score.
                texture_dna = replace(
                    plan.dna,
                    title=plan.dna.title + " / texture",
                    style="non-tonal atmospheric sound design, granular percussion, room texture, evolving noise",
                    instruments="metallic ticks, brushed noise, granular air, distant percussive texture",
                    production="wide diffuse texture, no dominant pitch, no sustained bass, leave space for foreground instruments",
                    mood=plan.dna.mood,
                    vocal_mode="instrumental",
                )
                texture_sections = [replace(s, direction=s.direction + "; texture bed only; avoid lead melody and sustained tones") for s in plan.sections]
                texture_plan = SongPlan(texture_dna, texture_sections, plan.segment_bars, plan.max_segment_seconds)
                bark_engine = BarkQuantumMusicEngine(self.kernel_cfg)
                mono = bark_engine.render(texture_plan, output_wav=str(Path(output_wav).with_name("_bark_texture_raw.wav")))
                bark_tex = self._fit_texture(mono, len(audio))
                # Keep Bark below the deterministic composition.
                bark_tex = bark_tex * float(np.clip(bark_texture_mix, 0.0, 0.35))
                audio = audio + bark_tex
                peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                if peak > 0.985:
                    audio *= 0.985 / peak

        write_stereo_wav(output_wav, self.synth.cfg.sample_rate, audio)
        if midi_path:
            try:
                export_midi(score, midi_path)
            except Exception as exc:
                print("MIDI export skipped:", exc)
        return HybridRenderResult(score, audio, self.synth.cfg.sample_rate, stems, bark_tex)


def musicality_report(score: SymbolicScore) -> Dict[str, Any]:
    """Cheap structural checks that detect 'held tones pretending to be music'."""
    lead = score.notes_for_part("lead")
    bass = score.notes_for_part("bass")
    unique_lead = len(set(n.midi for n in lead))
    onsets = sorted(set(round(n.start_beat, 3) for n in lead))
    durations = sorted(set(round(n.duration_beats, 3) for n in lead))
    return {
        "bars": score.total_bars,
        "seconds": score.seconds,
        "lead_notes": len(lead),
        "bass_notes": len(bass),
        "chord_or_arp_notes": len(score.notes_for_part("pad")) + len(score.notes_for_part("arp")),
        "drum_hits": len(score.drums),
        "unique_lead_pitches": unique_lead,
        "unique_lead_onsets": len(onsets),
        "lead_rhythm_values": durations,
        "motif_degrees": score.motif_degrees,
        "motif_rhythm": tuple(round(x, 3) for x in score.motif_rhythm),
        "passes_basic_music_structure": bool(len(lead) >= score.total_bars * 2 and unique_lead >= 4 and len(score.drums) >= score.total_bars * 4),
    }


if __name__ == "__main__":
    demo_dna = SongDNA(
        title="Nonlocal Circuit",
        bpm=122.0,
        root="D",
        mode="minor",
        style="glitch electronica with a memorable melodic hook",
        instruments="plucked lead, sub bass, arpeggiated harmony, electronic drums",
        mood="tense, luminous, propulsive",
    )
    demo_plan = SongPlan(demo_dna, [
        Section("intro", 4, "establish the motif with space", energy=0.38, mutation=0.08),
        Section("drive", 4, "bring in rhythmic momentum", energy=0.68, mutation=0.18),
        Section("lift", 4, "open the harmony and vary the hook", energy=0.82, mutation=0.34),
        Section("return", 4, "return to the hook and resolve", energy=0.62, mutation=0.10),
    ])
    engine = BarkScoreHybridEngine()
    result = engine.render(demo_plan, "bark_quantum_v3_demo.wav", "bark_quantum_v3_demo.mid", use_bark_texture=False)
    print(json.dumps(musicality_report(result.score), indent=2))

"""실습 11.3·11.4가 함께 쓰는 음악 도구.

이 파일은 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)가
직접 실행하지 않는다. 11-3, 11-4 실습 코드에서 import 해서 쓴다.
학생은 이 파일을 고치지 않아도 된다.

담은 것
- 음표 글자(C4:1)와 MIDI 음 번호 사이의 변환
- 멜로디 읽기: .txt(음표 글자)와 .mid(MIDI 파일) 둘 다
- 저장: WAV 소리 파일, MIDI 파일, 음높이 그림
- 모델: 바로 앞 음만 보는 빈도표, numpy로 만든 한 층짜리 LSTM
추가 패키지 없이 numpy, matplotlib, 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _krfont

NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TEMPO_BPM = 120          # 1박 = 0.5초
SAMPLE_RATE = 22050
GRID = 0.25              # MIDI를 읽을 때 박자를 맞추는 가장 작은 단위(16분음표)


# ---------------------------------------------------------------------------
# 음표 글자 <-> 음 번호
# ---------------------------------------------------------------------------
def parse_token(token: str) -> tuple[int | None, float]:
    """'E4:0.5' 같은 음표를 (MIDI 음 번호, 박자)로 바꾼다. 쉼표(R)는 음 번호가 None이다."""
    try:
        name, beats_text = token.split(":")
        beats = float(beats_text)
    except ValueError as exc:
        raise ValueError(f"음표 형식이 틀렸습니다: {token!r} (예: C4:1)") from exc
    if beats <= 0:
        raise ValueError(f"박자는 0보다 커야 합니다: {token!r}")
    if name == "R":
        return None, beats
    letter, rest = name[0], name[1:]
    if letter not in NOTE_OFFSETS or not rest:
        raise ValueError(f"음이름을 읽을 수 없습니다: {token!r}")
    shift = 0
    while rest and rest[0] in "#b":
        shift += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    if not rest.lstrip("-").isdigit():
        raise ValueError(f"옥타브 숫자가 없습니다: {token!r}")
    return 12 * (int(rest) + 1) + NOTE_OFFSETS[letter] + shift, beats


def make_token(midi: int | None, beats: float) -> str:
    beats_text = f"{beats:g}"
    if midi is None:
        return f"R:{beats_text}"
    return f"{SHARP_NAMES[midi % 12]}{midi // 12 - 1}:{beats_text}"


# ---------------------------------------------------------------------------
# 멜로디 읽기
# ---------------------------------------------------------------------------
def read_txt(path: Path) -> list[list[str]]:
    """한 줄에 한 곡. '#'으로 시작하는 줄은 설명이라 건너뛴다."""
    songs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        for token in tokens:
            try:
                parse_token(token)
            except ValueError as exc:
                raise ValueError(f"{path.name} {line_no}번째 줄: {exc}") from exc
        songs.append(tokens)
    return songs


def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def read_midi(path: Path) -> list[str]:
    """MIDI 파일에서 멜로디 한 줄을 뽑는다.

    여러 음이 동시에 울리면 가장 높은 음을 멜로디로 본다. 드럼 채널(10번)은 뺀다.
    박자는 16분음표 단위로 맞춘다.
    """
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"MIDI 파일이 아닙니다: {path.name}")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, n_tracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError(f"시간 단위가 SMPTE인 MIDI는 읽지 못합니다: {path.name}")
    pos = 8 + header_len

    notes: dict[int, tuple[int, int]] = {}   # 시작 틱 -> (가장 높은 음, 끝 틱)
    for _ in range(n_tracks):
        if data[pos:pos + 4] != b"MTrk":
            break
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        pos, end = pos + 8, pos + 8 + length
        tick, status = 0, 0
        sounding: dict[tuple[int, int], int] = {}   # (채널, 음) -> 시작 틱
        while pos < end:
            delta, pos = _read_varlen(data, pos)
            tick += delta
            if data[pos] & 0x80:
                status = data[pos]
                pos += 1
            if status == 0xFF:                       # 메타 이벤트
                pos += 1
                size, pos = _read_varlen(data, pos)
                pos += size
            elif status in (0xF0, 0xF7):             # 시스템 메시지
                size, pos = _read_varlen(data, pos)
                pos += size
            else:
                kind, channel = status & 0xF0, status & 0x0F
                n_bytes = 1 if kind in (0xC0, 0xD0) else 2
                args = data[pos:pos + n_bytes]
                pos += n_bytes
                if channel == 9 or kind not in (0x80, 0x90):
                    continue                         # 드럼과 음표 아닌 메시지는 건너뛴다
                key = (channel, args[0])
                if kind == 0x90 and args[1] > 0:
                    sounding[key] = tick
                elif key in sounding:                # 음 끄기
                    start = sounding.pop(key)
                    if start not in notes or args[0] > notes[start][0]:
                        notes[start] = (args[0], tick)
        pos = end

    if not notes:
        raise ValueError(f"음표가 없는 MIDI 파일입니다: {path.name}")

    def to_beats(ticks: int) -> float:
        return min(4.0, round(ticks / division / GRID) * GRID)

    starts = sorted(notes)
    tokens = []
    for k, start in enumerate(starts):
        pitch, stop = notes[start]
        nxt = starts[k + 1] if k + 1 < len(starts) else stop
        beats = to_beats(min(stop, nxt) - start)
        if beats >= GRID:
            tokens.append(make_token(pitch, beats))
        rest = to_beats(nxt - max(stop, start))
        if rest >= GRID:
            tokens.append(make_token(None, rest))
    return tokens


def load_folder_or_file(path: Path) -> tuple[list[list[str]], list[str]]:
    """파일 하나 또는 폴더 안의 .txt·.mid 파일을 모두 읽는다. (곡 목록, 곡 이름 목록)"""
    if not path.exists():
        raise FileNotFoundError(f"멜로디 파일이나 폴더가 없습니다: {path}")
    files = [path] if path.is_file() else sorted(
        p for p in path.iterdir() if p.suffix.lower() in (".txt", ".mid", ".midi"))
    songs, names = [], []
    for f in files:
        if f.suffix.lower() == ".txt":
            found = read_txt(f)
            songs += found
            names += [f"{f.name} {k + 1}번째 줄" for k in range(len(found))]
        else:
            songs.append(read_midi(f))
            names.append(f.name)
    if not songs:
        raise ValueError(f"읽을 멜로디가 하나도 없습니다: {path}")
    return songs, names


# ---------------------------------------------------------------------------
# 저장: WAV, MIDI, 그림
# ---------------------------------------------------------------------------
def write_wav(tokens: list[str], path: Path) -> float:
    """사인파 소리로 WAV 파일을 만들고 길이(초)를 돌려준다."""
    sec_per_beat = 60.0 / TEMPO_BPM
    pieces = []
    for token in tokens:
        midi, beats = parse_token(token)
        n = int(SAMPLE_RATE * beats * sec_per_beat)
        t = np.arange(n) / SAMPLE_RATE
        if midi is None:
            pieces.append(np.zeros(n))
            continue
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        envelope = np.minimum(1.0, t / 0.01) * np.exp(-3.0 * t)  # 건반을 친 듯 줄어드는 소리
        pieces.append(0.4 * envelope * np.sin(2 * np.pi * freq * t))
    audio = np.concatenate(pieces)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return len(audio) / SAMPLE_RATE


def _varlen(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi(tokens: list[str], path: Path, ticks_per_beat: int = 480) -> None:
    """피아노 한 대가 연주하는 MIDI 파일로 저장한다. 작곡 프로그램에서 열어 고칠 수 있다."""
    events = bytearray()
    events += _varlen(0) + bytes([0xFF, 0x51, 0x03]) + int(60_000_000 / TEMPO_BPM).to_bytes(3, "big")
    events += _varlen(0) + bytes([0xC0, 0x00])          # 악기: 피아노
    wait = 0
    for token in tokens:
        midi, beats = parse_token(token)
        ticks = int(round(beats * ticks_per_beat))
        if midi is None:
            wait += ticks
            continue
        events += _varlen(wait) + bytes([0x90, midi, 90])
        events += _varlen(ticks) + bytes([0x80, midi, 0])
        wait = 0
    events += _varlen(wait) + bytes([0xFF, 0x2F, 0x00])
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    path.write_bytes(header + b"MTrk" + struct.pack(">I", len(events)) + bytes(events))


def draw_rolls(melodies: dict[str, list[str]], prompt_len: int, out_path: Path,
               ko: bool, losses: list[float] | None = None) -> None:
    """멜로디를 음높이 그림으로 그린다. 가로 = 박, 세로 = 음높이."""
    L = _krfont.label
    rows = len(melodies) + (1 if losses else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.3 * rows), squeeze=False)
    axes = axes[:, 0]
    if losses:
        axes[0].plot(losses, color="#4C78A8")
        axes[0].set_title(L("LSTM 학습 손실 (낮을수록 다음 음을 잘 맞힌다)",
                            "LSTM training loss", ko), fontsize=11)
        axes[0].set_xlabel("epoch")
        axes = axes[1:]
    for ax, (name, tokens) in zip(axes, melodies.items()):
        time = 0.0
        for k, token in enumerate(tokens):
            midi, beats = parse_token(token)
            if midi is not None:
                color = "#79706E" if k < prompt_len else "#F58518"
                ax.barh(midi, beats * 0.95, left=time, height=0.8, color=color)
            time += beats
        ax.set_title(name, fontsize=11)
        ax.set_ylabel(L("음높이", "pitch", ko))
        ax.yaxis.get_major_locator().set_params(integer=True)
    axes[-1].set_xlabel(L("박", "beats", ko))
    fig.suptitle(L("회색 = 시작 멜로디, 주황 = 모델이 이어서 만든 음",
                   "gray = prompt, orange = generated", ko), fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def build_vocab(songs: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    vocab = sorted({tok for song in songs for tok in song},
                   key=lambda t: (parse_token(t)[0] or 0, parse_token(t)[1]))
    return vocab, {tok: i for i, tok in enumerate(vocab)}


def bigram_table(songs_idx: list[list[int]], vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size))
    for song in songs_idx:
        for prev, nxt in zip(song[:-1], song[1:]):
            counts[prev, nxt] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)


def generate_bigram(table: np.ndarray, prompt: list[int], n: int, temperature: float,
                    rng: np.random.Generator) -> list[int]:
    out = list(prompt)
    for _ in range(n):
        row = table[out[-1]]
        if row.sum() == 0:
            break  # 이 음 뒤에 온 음이 학습 곡에 없다
        p = np.where(row > 0, row ** (1.0 / temperature), 0.0)
        out.append(int(rng.choice(len(row), p=p / p.sum())))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TinyLSTM:
    """한 층짜리 LSTM. 입력은 음표의 원-핫 벡터, 출력은 다음 음표의 확률이다."""

    def __init__(self, vocab_size: int, hidden: int, rng: np.random.Generator):
        self.V, self.H = vocab_size, hidden
        scale = 1.0 / np.sqrt(vocab_size + hidden)
        self.p = {
            "W": rng.normal(0, scale, (4 * hidden, vocab_size + hidden)),  # 네 개의 문(gate)
            "b": np.zeros(4 * hidden),
            "Wy": rng.normal(0, 1.0 / np.sqrt(hidden), (vocab_size, hidden)),
            "by": np.zeros(vocab_size),
        }
        self.p["b"][hidden:2 * hidden] = 1.0  # 처음에는 잊지 않는 쪽으로 시작한다
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam 기록
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def step(self, x_idx: int, h: np.ndarray, c: np.ndarray):
        """음표 하나를 읽고 기억(h, c)을 갱신한다."""
        H = self.H
        x = np.zeros(self.V)
        x[x_idx] = 1.0
        z = np.concatenate([x, h])
        a = self.p["W"] @ z + self.p["b"]
        i, f, o = sigmoid(a[:H]), sigmoid(a[H:2 * H]), sigmoid(a[2 * H:3 * H])
        g = np.tanh(a[3 * H:])
        c_new = f * c + i * g          # 기억 칸: 일부는 잊고(f) 새것을 더한다(i*g)
        h_new = o * np.tanh(c_new)     # 밖으로 내보낼 기억
        logits = self.p["Wy"] @ h_new + self.p["by"]
        return h_new, c_new, logits, (z, i, f, o, g, c, c_new)

    def loss_and_grads(self, seq: list[int]):
        """곡 하나에서 '다음 음표 맞히기' 손실과 기울기를 구한다."""
        H = self.H
        h, c = np.zeros(H), np.zeros(H)
        caches, probs, hs = [], [], []
        loss = 0.0
        for t in range(len(seq) - 1):
            h, c, logits, cache = self.step(seq[t], h, c)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            loss -= np.log(p[seq[t + 1]] + 1e-12)
            caches.append(cache)
            probs.append(p)
            hs.append(h)

        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh_next, dc_next = np.zeros(H), np.zeros(H)
        for t in reversed(range(len(seq) - 1)):
            z, i, f, o, g, c_prev, c_t = caches[t]
            dlogits = probs[t].copy()
            dlogits[seq[t + 1]] -= 1.0
            grads["Wy"] += np.outer(dlogits, hs[t])
            grads["by"] += dlogits
            dh = self.p["Wy"].T @ dlogits + dh_next
            tanh_c = np.tanh(c_t)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            di, df, dg = dc * g, dc * c_prev, dc * i
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - g ** 2)])
            grads["W"] += np.outer(da, z)
            grads["b"] += da
            dz = self.p["W"].T @ da
            dh_next = dz[self.V:]
            dc_next = dc * f
        n = max(1, len(seq) - 1)
        return loss / n, {k: v / n for k, v in grads.items()}

    def update(self, grads: dict, lr: float) -> None:
        """Adam으로 가중치를 조금 고친다. 기울기가 너무 크면 잘라 낸다."""
        self.t += 1
        for k in self.p:
            gk = np.clip(grads[k], -5, 5)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            m_hat = self.m[k] / (1 - 0.9 ** self.t)
            v_hat = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def train(self, songs_idx: list[list[int]], epochs: int, lr: float,
              rng: np.random.Generator, report_every: int = 20) -> list[float]:
        losses = []
        for epoch in range(1, epochs + 1):
            total = 0.0
            for k in rng.permutation(len(songs_idx)):
                loss, grads = self.loss_and_grads(songs_idx[k])
                self.update(grads, lr)
                total += loss
            losses.append(total / len(songs_idx))
            if epoch == 1 or epoch % report_every == 0:
                print(f"  epoch {epoch:4d}  손실 {losses[-1]:.3f}")
        return losses

    def generate(self, prompt: list[int], n: int, temperature: float,
                 rng: np.random.Generator) -> list[int]:
        """시작 멜로디를 읽은 뒤, 다음 음표를 하나씩 뽑아 이어 붙인다."""
        h, c = np.zeros(self.H), np.zeros(self.H)
        for idx in prompt[:-1]:
            h, c, _, _ = self.step(idx, h, c)
        out, last = list(prompt), prompt[-1]
        for _ in range(n):
            h, c, logits, _ = self.step(last, h, c)
            scaled = logits / temperature
            p = np.exp(scaled - scaled.max())
            p /= p.sum()
            last = int(rng.choice(self.V, p=p))
            out.append(last)
        return out


def copied_ratio(made: list[str], songs: list[list[str]], n: int = 4) -> float:
    """만든 곡의 '연속 n음' 가운데 학습 곡에 그대로 있던 것의 비율(0~1)."""
    seen = {tuple(s[i:i + n]) for s in songs for i in range(len(s) - n + 1)}
    grams = [tuple(made[i:i + n]) for i in range(len(made) - n + 1)]
    return sum(g in seen for g in grams) / len(grams) if grams else 0.0


def longest_copy(made: list[str], songs: list[list[str]]) -> int:
    """만든 곡 안에서 학습 곡 하나와 그대로 겹치는 가장 긴 연속 구간의 음표 수."""
    best = 0
    for song in songs:
        for i in range(len(made)):
            for j in range(len(song)):
                k = 0
                while i + k < len(made) and j + k < len(song) and made[i + k] == song[j + k]:
                    k += 1
                best = max(best, k)
    return best

"""실습 11.3 - 내 멜로디로 LSTM을 학습시켜 새 곡을 만든다.

언어모델이 "다음 단어"를 고르듯, 이 코드는 "다음 음표"를 고른다.
진행 순서는 네 단계다.

  1. data/my_music/ 폴더의 곡을 모두 읽는다 (.txt 음표 글자, .mid MIDI 파일)
  2. LSTM에게 "다음 음표 맞히기"를 반복시켜 학습한다
  3. 온도를 바꿔 가며 후보 곡을 여러 개 만든다
  4. 후보마다 WAV(듣기용), MIDI(편곡용), TXT(음표 목록)를 저장하고,
     학습 곡을 얼마나 그대로 베꼈는지 표로 보여 준다

학생은 아래 "학생이 바꿔 볼 값"과 data/my_music/ 폴더만 고치면 된다.
외부 API와 추가 패키지를 쓰지 않는다(numpy, matplotlib, 표준 라이브러리).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

import _krfont
import _music

# ---------------------------------------------------------------------------
# 학생이 바꿔 볼 값 (여기만 고치면 된다)
# ---------------------------------------------------------------------------
MUSIC_DIR = "my_music"            # practice/chapter11/data/ 안의 학습 곡 폴더
PROMPT = ""                       # 시작 멜로디. 예: "C4:1 E4:1 G4:2". 비우면 첫 곡의 앞 4음
LENGTH = 48                       # 시작 멜로디 뒤에 이어 만들 음표 수
TEMPERATURES = [1.0, 1.5, 2.0]    # 후보를 만들 온도. 낮으면 원곡에 가깝고 높으면 새롭다
CANDIDATES_PER_T = 2              # 온도마다 만들 후보 수
EPOCHS = 80                       # 학습 반복 횟수. 늘리면 학습 곡을 외운다
HIDDEN = 48                       # LSTM 기억 칸의 개수
SEED = 2026                       # 바꾸면 다른 후보가 나온다
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.01
CHAPTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_DIR / "data"
OUT_DIR = CHAPTER_DIR / "results" / "my_song"


def main() -> None:
    ko = _krfont.setup()
    _krfont.report(ko)
    rng = np.random.default_rng(SEED)
    if OUT_DIR.exists():
        try:
            shutil.rmtree(OUT_DIR)        # 이전 실행의 후보를 지우고 새로 만든다
        except PermissionError as error:  # Windows는 다른 프로그램이 연 파일을 지우지 못한다
            print(f"이전 후보 파일을 지우지 못했다: {error.filename}")
            print("음악 재생기나 작곡 프로그램에서 results/my_song/ 의 파일을 닫고 다시 실행한다.")
            raise SystemExit(1) from error
    OUT_DIR.mkdir(parents=True)

    print("=== 실습 11.3  내 멜로디로 새 곡 만들기 ===")
    print(f"설정: MUSIC_DIR={MUSIC_DIR}, LENGTH={LENGTH}, TEMPERATURES={TEMPERATURES}, "
          f"CANDIDATES_PER_T={CANDIDATES_PER_T}, EPOCHS={EPOCHS}, HIDDEN={HIDDEN}, SEED={SEED}")
    print()

    # 1단계 ------------------------------------------------------------------
    songs, names = _music.load_folder_or_file(DATA_DIR / MUSIC_DIR)
    vocab, index = _music.build_vocab(songs)
    songs_idx = [[index[t] for t in song] for song in songs]
    print(f"[1] 학습 곡 읽기 - {len(songs)}곡, 음표 {sum(map(len, songs))}개, "
          f"서로 다른 음표 {len(vocab)}개")
    for name, song in zip(names, songs):
        print(f"  {name:<28} {len(song):3d}음  {' '.join(song[:6])} ...")
    print()

    # 2단계 ------------------------------------------------------------------
    print("[2] LSTM 학습 - 곡마다 '다음 음표 맞히기'를 반복한다")
    model = _music.TinyLSTM(len(vocab), HIDDEN, rng)
    losses = model.train(songs_idx, EPOCHS, LEARNING_RATE, rng)
    print()

    # 3단계 ------------------------------------------------------------------
    prompt_tokens = PROMPT.split() if PROMPT.strip() else songs[0][:4]
    missing = [t for t in prompt_tokens if t not in index]
    if missing:
        raise SystemExit(f"시작 멜로디의 {missing}는 학습 곡에 없는 음표입니다. "
                         f"쓸 수 있는 음표: {' '.join(vocab)}")
    prompt = [index[t] for t in prompt_tokens]
    print(f"[3] 후보 만들기 - 시작 멜로디: {' '.join(prompt_tokens)}")

    rows, rolls = [], {}
    number = 0
    for temperature in TEMPERATURES:
        for _ in range(CANDIDATES_PER_T):
            number += 1
            tokens = [vocab[i] for i in model.generate(prompt, LENGTH, temperature, rng)]
            stem = f"candidate_{number:02d}"
            seconds = _music.write_wav(tokens, OUT_DIR / f"{stem}.wav")
            _music.write_midi(tokens, OUT_DIR / f"{stem}.mid")
            (OUT_DIR / f"{stem}.txt").write_text(" ".join(tokens) + "\n", encoding="utf-8")
            ratio = _music.copied_ratio(tokens, songs)
            longest = _music.longest_copy(tokens, songs)
            rows.append([stem, temperature, f"{ratio * 100:.1f}", longest, f"{seconds:.1f}"])
            rolls[f"{stem}  (T={temperature}, 4음 일치 {ratio * 100:.0f}%)"] = tokens
    print(f"  후보 {len(rows)}개를 만들었다")
    print()

    # 4단계 ------------------------------------------------------------------
    print("[4] 후보 비교 - 학습 곡을 얼마나 그대로 따라갔는가")
    print("  후보            온도   4음 일치율   가장 긴 복사 구간   길이")
    for stem, t, ratio, longest, seconds in rows:
        print(f"  {stem:<14} {t:4.1f}   {ratio:>6}%     {longest:3d}음            {seconds}초")
    print("  4음 일치율: 후보의 연속 4음 가운데 학습 곡에 그대로 있던 비율.")
    print("  가장 긴 복사 구간: 학습 곡 한 곡과 음표가 그대로 이어지는 가장 긴 길이.")
    print()

    with open(OUT_DIR / "candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["후보", "온도", "4음 일치율(%)", "가장 긴 복사 구간(음)", "길이(초)"])
        writer.writerows(rows)
    _music.draw_rolls(rolls, len(prompt), OUT_DIR / "candidates.png", ko, losses)
    print(f"저장: results/my_song/ 에 후보 {len(rows)}개 × (wav, mid, txt), "
          f"candidates.csv, candidates.png")


if __name__ == "__main__":
    main()

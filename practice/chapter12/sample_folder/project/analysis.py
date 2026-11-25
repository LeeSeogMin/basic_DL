"""설문 응답 평균을 낸다 (합성 예제, 2판)."""
import pandas as pd

raw = pd.read_csv("data/responses_raw.csv")
clean = raw.dropna().copy()                    # 빈 칸이 있는 응답은 분석에서만 뺀다. 원본은 고치지 않는다
clean["focus"] = clean[["q1", "q2", "q3", "q4"]].mean(axis=1)
print(clean.groupby(clean["sleep_hours"] >= 7)["focus"].mean())

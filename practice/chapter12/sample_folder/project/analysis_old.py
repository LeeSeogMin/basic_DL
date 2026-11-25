"""예전 분석 코드 (합성 예제, 1판 설문용). 2판으로 바뀌어 지금은 쓰지 않는다."""
import pandas as pd

raw = pd.read_csv("data/responses_raw.csv")
print(raw.mean(numeric_only=True))

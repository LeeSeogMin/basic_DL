# 아직 다 쓰지 않은 코드다. 문법 오류를 일부러 남겨 두었다.
# 실습 14.2의 점검 도구가 이 파일을 "문법 오류"로 표시한다.
# 이 파일은 교재용 예시이므로 고치지 않는다.


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8")
        rows.append(line.strip())
    return rows

"""합성 예제 파일. 실습 12.1이 확장자를 세는 대상일 뿐, 실행하지 않는다."""


def add(a, b):
    return a + b


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

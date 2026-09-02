# 무료 코딩 에이전트 설치 가이드

> GitHub Copilot은 **VS Code**에서 사용하고, Antigravity와 TRAE는 별도 앱(IDE/워크스테이션)으로 실행한다.

---

## Git과 GitHub 먼저 이해하기

Git과 GitHub는 같은 프로그램이 아니다.

- **Git**: 내 컴퓨터에서 파일 변경 이력을 기록하고, 이전 상태와 비교하는 버전 관리 도구다.
- **GitHub**: Git 저장소를 인터넷에 저장하고, 공유·협업·제출에 사용하는 웹 서비스다.
- Git은 컴퓨터에 설치하지만, GitHub는 웹 브라우저에서 계정을 만든 뒤 사용한다. GitHub Desktop을 쓰려면 별도 앱을 설치할 수 있다.

### Windows에서 Git 설치하기

1. [Git for Windows 공식 설치 페이지](https://git-scm.com/install/windows)를 연다.
2. Windows용 설치 파일을 내려받아 실행한다.
3. 설치 과정의 기본 설정을 유지하고 설치를 끝낸다.
4. PowerShell 또는 VS Code 터미널을 새로 열고 아래 명령으로 설치를 확인한다.

```powershell
git --version
```

버전이 출력되면 Git 설치가 끝났다. 처음 한 번은 아래처럼 Git에 사용할 이름과 이메일을 등록한다. 이메일은 GitHub 계정에 등록한 이메일을 사용한다.

```powershell
git config --global user.name "내 이름"
git config --global user.email "내 이메일"
```

### GitHub 계정 만들기

1. [GitHub 가입 페이지](https://github.com/signup)를 연다.
2. 개인 계정을 만들고 이메일 인증을 끝낸다.
3. 과제 저장소를 공유할 때 사용할 사용자 이름을 확인한다.
4. 계정 보안을 위해 2단계 인증을 설정한다.

GitHub 저장소를 컴퓨터와 연결할 때는 Git이 필요하다. 수업에서는 먼저 Git으로 변경 사항을 기록한 뒤 GitHub에 저장소를 올리는 흐름을 사용한다.

---

## 1. 설치 전 공통 준비물

먼저 아래 항목을 준비한다. 인터넷 연결과 프로그램 설치 권한은 네 도구에 모두 필요하다. 나머지는 사용할 도구에 따라 준비한다.

### 네 도구에 공통으로 필요한 항목

| 항목 | 준비 방법 | 확인 방법 |
|------|----------|----------|
| 인터넷 연결 | 웹사이트 접속과 로그인, 프로그램 다운로드가 가능한 네트워크 사용 | 브라우저에서 설치 페이지 접속 |
| 웹 브라우저 | Edge, Chrome, Firefox 등 설치 | 로그인 페이지가 열리는지 확인 |
| 프로그램 설치 권한 | 개인 PC 또는 프로그램을 설치할 수 있는 실습 PC 사용 | 설치 파일 실행 가능 여부 확인 |
| 프로젝트 작업 폴더 | 파일을 저장하고 수정할 권한이 있는 폴더 준비 | VS Code, Antigravity 또는 TRAE에서 폴더 열기 |
| 프로젝트 규칙 파일 | 작업 폴더 루트에 `AGENTS.md` 작성 (5절) | 도구에 "이 프로젝트 규칙을 요약해줘"라고 물어 확인 |

### 도구별로 필요한 항목

| 사용할 도구 | 추가 준비물 | 설치·가입 방법 | 확인 방법 |
|------------|------------|---------------|----------|
| GitHub Copilot | VS Code, 개인 GitHub 계정, 학생 신분 증빙 | [VS Code](https://code.visualstudio.com) 설치, [GitHub](https://github.com) 가입 | VS Code 실행, GitHub 로그인 |
| Antigravity | Google 계정 | [Google 계정](https://accounts.google.com) 준비 | 브라우저 로그인 |
| TRAE | TraeWork(또는 TraeCode) 데스크톱 앱 | [TRAE 개요](https://docs.trae.cn/) 참고 후 설치 | 앱 실행 후 프로젝트 폴더 열기 |

Git은 설치 자체의 필수 조건은 아니지만, 수업 저장소를 내려받고 변경 이력을 관리하려면 설치하는 편이 낫다. [git-scm.com](https://git-scm.com)에서 설치한 뒤 `git --version`으로 확인한다.

---

## 2. GitHub Copilot Student (신청 → 승인 → VS Code 설치)

### 2-1. 왜 까다로운가

- GitHub Education Student Developer Pack을 먼저 승인받아야 Copilot을 쓸 수 있다
- 학생 신분 증명을 통과해야 하고, 승인까지 수 시간~2-3일이 걸린다
- 서류가 불분명하면 거절되고 다시 신청해야 한다
- 2026년 4-6월에 신규 가입이 일시 중단된 적이 있다. 2026년 6월 17일부터 재개되어 현재 신규 가입이 가능하다. 다만 정책이 바뀔 수 있으므로 신청 페이지의 최신 공지를 확인한다

### 2-2. 자격 조건

- 학위·졸업장 수여 과정에 재학 중
- 만 13세 이상
- **개인 GitHub 계정** 보유 (조직 계정 불가)

### 2-3. 신청 절차 (단계별)

**① GitHub 계정 준비**
- [github.com](https://github.com)에서 개인 계정을 만든다 (이미 있으면 건너뜀)
- 프로필 이름을 실명으로 설정한다 — 서류의 이름과 일치해야 한다

**② Student Developer Pack 신청**
1. [education.github.com/pack](https://education.github.com/pack) 접속
2. **"Sign up for Student Developer Pack"** 또는 **"Get benefits"** 클릭
3. 역할 선택: **Student**
4. 학교 선택: 학교 이름을 검색하거나 직접 입력

**③ 학생 신분 증명 (2가지 방법)**

○ **방법 A: 학교 이메일 인증 (가장 빠름)**
  - `@university.ac.kr`, `@univ.edu` 같은 학교 이메일을 GitHub 계정에 추가한다
  - Settings → Emails → 학교 이메일 추가 → 인증 메일 확인
  - 신청 시 학교 이메일을 선택하면 자동으로 통과하는 경우가 많다

○ **방법 B: 서류 업로드 (학교 이메일이 없거나 인증이 안 될 때)**

| 서류 종류 | 영문 표기 | 주의사항 |
|----------|----------|---------|
| 학생증 사진 | Student ID | 이름 + 학교명 + **현재 재학 기간/날짜**가 보여야 한다 |
| 수업 시간표 | Class schedule | 이름, 학교명, 현재 학기 수업이 표시되어야 한다 |
| 성적 증명서 | Transcript | 이름과 현재 재학 상태가 보여야 한다 |
| 재학증명서 | Enrollment verification letter | 학교에서 발급한 공식 서류. 영문이면 더 좋다 |
| 등록금 납부 영수증 | Tuition receipt | 현재 학기 납부 내역 (이름·학교·날짜 필수) |
| 학교 포털 스크린샷 | Screenshot of student portal | 본인 이름 + 현재 재학 상태가 보이는 화면 캡처 |

**서류 업로드 시 거절을 피하려면:**
- 글자가 선명하게 보여야 한다 — 흐리거나 잘리면 거절된다
- GitHub 프로필 이름과 서류의 이름이 일치해야 한다
- 현재 학기/연도가 보여야 한다 — 오래된 서류는 거절된다
- 이미지 파일로 올린다 — PDF는 지원이 안된다. 즉 위의 서류를 받으면 png 파일 등으로 chatgpt 등에서 변환한다. 

**④ 승인 대기**
- 보통 수 시간 ~ 2-3일
- 거절되면 이메일에 이유가 오고, 다른 서류로 다시 신청할 수 있다

### 2-4. Copilot Student 혜택 (승인 후)

| 항목 | 내용 |
|------|------|
| 코드 완성 | 무제한 |
| AI Credits | 월 200 (채팅·에이전트용) |
| 모델 선택 | **Auto만 가능** (수동 선택 불가) |

### 2-5. VS Code에서 Copilot 활성화

1. VS Code를 연다
2. 확장(Extensions) 탭에서 **"GitHub Copilot"** 검색 → 설치
3. 좌측 하단 사람 아이콘 → **Sign in with GitHub** → 로그인
4. 상태 바에 Copilot 아이콘이 나타나면 활성화된 것이다
5. 확인: [github.com/settings/copilot](https://github.com/settings/copilot) 에서 Copilot Student 상태 확인

---

## 3. Antigravity IDE (Google)

### 3-1. 설치 (Windows)

1. [antigravity.google/download](https://antigravity.google/download)에서 **Windows** 버전 다운로드 (x64)
2. 다운로드한 `.exe` 실행
3. Windows Defender SmartScreen이 뜨면 **"추가 정보" → "실행"** 클릭
4. 설치 완료 후 실행 → **Google 계정으로 로그인**
5. 테마 선택 및 에이전트 정책(터미널 실행, 코드 리뷰 등) 설정

### 3-2. 현재 버전 (2026년 8월 기준)

| 항목 | 버전 |
|------|------|
| Antigravity 2.0 | v2.8.1 |
| Antigravity IDE | v2.5.5 |
| 지원 OS | Windows 10/11 64-bit |

### 3-3. VS Code와의 관계

- Antigravity는 **독립 IDE**이므로 VS Code 안에서 실행하는 것이 아니다
- VS Code 프로젝트와 같은 폴더를 열어 병행 사용할 수 있다
- Gemini 기반이라 Google 계정만 있으면 별도 API 키 없이 바로 쓸 수 있다

### 3-4. CLI 설치 (선택)

PowerShell에서:
```powershell
irm https://antigravity.google/cli/install.ps1 | iex
```

---

## 4. TRAE (TraeWork / TraeCode)

> 참고: TRAE는 IDE( TraeCode )와 AI 워크스테이션( TraeWork ) 등 여러 제품군으로 구성된다. 상황에 따라 하나만 설치해도 된다. ([TRAE 개요](https://docs.trae.cn/))

### 4-1. 무엇을 할 수 있나

- 자연어로 목표를 주면, 작업을 쪼개서 계획하고(Plan) 실행(Build)하는 **에이전트(Agent)** 중심의 흐름을 제공한다
- 코드뿐 아니라 문서/리포트 같은 산출물 생성, 프로젝트 맥락 기반의 수정 작업에 적합하다

### 4-2. 설치 (Windows)

1. 공식 문서의 다운로드 안내에서 TraeWork(또는 TraeCode) 데스크톱 앱을 설치한다: <https://docs.trae.cn/>
2. 앱을 실행한 뒤, 과제/프로젝트 폴더를 연다
3. Code/IDE 모드에서 채팅 또는 Agent 기능으로 작업을 진행한다
4. $3, $10 등 저렴한 가격도 있다. 

### 4-3. 처음 사용할 때 팁

- 처음엔 “현재 폴더에서 `README`를 읽고 해야 할 일을 정리해줘” 같은 작은 요청부터 시작한다
- 계획(Plan)과 실행(Build)이 나뉘는 흐름이 있으면, 실행 전에 계획을 먼저 확인한다

---

## 5. 프로젝트 규칙 파일 설정 (AGENTS.md · CLAUDE.md · context.md · todo.md)

에이전트가 프로젝트 규칙과 진행 상황을 이어받도록 작업 폴더 루트에 관련 파일을 둔다.

| 파일 | 역할 | 읽히는 방식 |
|------|------|-------------|
| `AGENTS.md` | 실행 명령, 폴더 구조, 코드 규칙, 금지 사항 | 대부분의 코딩 에이전트가 자동으로 읽음 |
| `CLAUDE.md` | Claude Code용 규칙 파일 | Claude Code가 자동으로 읽음 |
| `context.md` | 현재 상태와 주요 결정 | `AGENTS.md`에서 읽도록 지정 |
| `todo.md` | 남은 작업과 완료 항목 | `AGENTS.md`에서 읽도록 지정 |

규칙은 `AGENTS.md`를 원본으로 둔다. Claude Code를 함께 쓰면 같은 폴더에 `CLAUDE.md`를 만들고 다음처럼 연결한다.

```markdown
@AGENTS.md

작업 시작 전 context.md와 todo.md를 읽는다.
```

최소 구성은 다음과 같다.

```text
my-project/
├── AGENTS.md
├── CLAUDE.md
├── context.md
└── todo.md
```

설정 후 에이전트에 **“이 프로젝트 규칙을 요약해줘”**라고 요청한다. `AGENTS.md`의 내용이 답에 나오면 규칙 파일을 읽은 것이다.

---

부록: 

### 1. 교수자의 코드에서 강의자료 클론하기

- vscode에서 폴더 열기로 내 컴퓨터 내 문서 폴더 열기

- https://github.com/LeeSeogMin/basic_dl.git을 현재 폴더에서 클론해줘.

### 2. 내가 작업한 것은 그대로 남기고 교수자의 변경 강의자료 가져오기

- 1단계: 내가 수정한 실습 코드를 임시 보관함에 안전하게 저장하기
git stash

- 2단계: 강사 최신 업데이트 자료 다운로드하기
git pull origin main

- 3단계: 임시 보관함에 넣어둔 내 코드 다시 꺼내와서 합치기
git stash pop
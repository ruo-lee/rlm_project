# RLM: Recursive Language Model

Gemini 기반 Recursive Language Model 구현. **동적 파일 접근** 아키텍처로 대용량 프로젝트 분석 가능.

## 설치

```bash
uv sync
cp .env.local.example .env.local
# GEMINI_API_KEY 설정
```

## 사용법

### 🖥️ TUI 모드
```bash
python -m rlm
```

**슬래시 명령어:**
| 명령어 | 설명 |
|--------|------|
| `/project` | 프로젝트 목록 보기 |
| `/project <N>` | 프로젝트 N 선택 |
| `/model` | 사용 가능한 모델 목록 |
| `/model <name>` | 모델 변경 |
| `/help` | 도움말 |
| `/clear` | 채팅 초기화 |

**단축키:**
| 키 | 동작 |
|----|------|
| `Ctrl+P` | 명령어 팔레트 |
| `Ctrl+L` | 채팅 초기화 |
| `Ctrl+Q` | 종료 |

### 📊 벤치마크 (모델 비교)
```bash
# 모델 비교
python -m rlm.cli.benchmark -p 1 -q "질문" -m "gemini-3-flash-preview,gemini-2.5-flash"

# 결과 JSON 저장
python -m rlm.cli.benchmark -p 1 -q "질문" -o results.json

# 프로젝트 목록
python -m rlm.cli.benchmark --list
```

## 환경변수

`.env.local`:
```bash
# 필수
GEMINI_API_KEY=your_api_key

# 기본 모델
GEMINI_MODEL_NAME=gemini-3-flash-preview

# 모델 목록 커스텀 (선택)
RLM_AVAILABLE_MODELS=gemini-3-flash-preview,gemini-2.5-flash,gemini-2.5-pro
```

## 동적 파일 접근 (핵심 기능)

LLM이 **REPL 도구**를 통해 프로젝트 파일을 동적으로 탐색:

| 도구 | 설명 |
|------|------|
| `list_files()` | 프로젝트 내 파일 목록 |
| `read_file(name, start, max)` | 파일 내용 읽기 (라인 범위) |
| `search_files(keyword)` | 키워드 검색 |
| `get_file_info(name)` | 파일 메타정보 |

**지원 포맷:** PDF, DOCX, PPTX, TXT, MD, CSV, JSON, XML, PY, JS, HTML, CSS 등

## 프로젝트 추가

`data/projects/` 폴더에 하위 폴더 생성:

```
data/projects/
├── 내프로젝트/          # 자동 인식
│   ├── doc1.pdf
│   ├── doc2.docx
│   └── notes.txt
└── 법률문서/
    └── contract.pdf
```

## 프로젝트 구조

```
rlm_project/
├── rlm/
│   ├── cli/
│   │   ├── main.py         # TUI 런처
│   │   └── benchmark.py    # 벤치마크 CLI
│   ├── tui/
│   │   ├── app.py          # TUI 앱
│   │   └── styles.tcss     # 스타일
│   ├── core/
│   │   ├── agent.py        # RLM 에이전트
│   │   └── config.py       # 설정
│   ├── repl/
│   │   └── executor.py     # REPL + 파일 도구
│   ├── llm/
│   │   └── client.py       # Gemini 클라이언트
│   ├── data/
│   │   └── datasets.py     # 프로젝트 로더
│   └── parsers/
│       └── loader.py       # PDF/DOCX/PPTX 파서
│
├── data/projects/          # 프로젝트 폴더들
└── logs/                   # 로그 파일
```

## 라이선스

MIT

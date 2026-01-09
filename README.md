# RLM: Recursive Language Models

Gemini 기반 Recursive Language Model 구현 및 벤치마크.

## 설치

```bash
uv sync
cp .env.local.example .env.local
# GEMINI_API_KEY 설정
```

## 사용법

### 기본 실행 (대화형)
```bash
uv run src/main.py
```

### CLI 모드
```bash
uv run src/main.py -q 1              # 쿼리 1번 실행
uv run src/main.py -q 3 -s 500k      # 쿼리 3번, 500K 컨텍스트
uv run src/main.py --query "커스텀 질문"
uv run src/main.py --list            # 쿼리 목록
uv run src/main.py --sandbox         # 보안 샌드박스 모드
```

### 벤치마크 (Baseline vs Optimized)
```bash
uv run src/benchmark.py -q "긍정 단어 찾기" -s 100k
uv run src/benchmark.py --baseline-only
uv run src/benchmark.py -o results.json
```

## 구조

```
src/
├── main.py          # CLI 진입점
├── rlm.py           # 최적화 RLM (캐싱, 병렬)
├── rlm/
│   ├── baseline.py  # 원 논문 방식
│   └── config.py    # 가격 설정
├── repl.py          # 최적화 REPL
└── benchmark.py     # 비교 유틸리티
```

## 개선 사항 (vs 원 논문)

| 기능 | Baseline | Optimized |
|------|----------|-----------|
| llm_query | 순차 | 캐싱 |
| llm_query_batch | ❌ | ✅ 병렬 |
| RecursionGuard | ❌ | ✅ |
| 비용 추적 | ❌ | ✅ |

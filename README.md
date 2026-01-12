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
uv run src/main.py -d 3 -q 8         # Law Insider 데이터셋, 쿼리 8번
uv run src/main.py --query "커스텀 질문"
uv run src/main.py --list            # 쿼리 목록
uv run src/main.py --list-datasets   # 데이터셋 목록
uv run src/main.py --sandbox         # 보안 샌드박스 모드
```

### 벤치마크 (Baseline vs Optimized)
```bash
uv run src/main.py --benchmark -q 1 -d 1        # 전체 비교
uv run src/main.py --benchmark --baseline -q 1  # Baseline만
uv run src/main.py --benchmark --optimized -q 8 -d 3  # Optimized만
uv run src/main.py --benchmark -q 1 -o results.json   # 결과 저장
```

### Law Insider 데이터셋 (PDF/DOCX 추출)
```bash
uv run src/extract_documents.py      # PDF/DOCX → 텍스트 추출
uv run src/main.py -d 3 -q 9 -s 100k # 정의 조항 찾기
```

## 구조

```
src/
├── main.py              # 통합 CLI 진입점 (벤치마크 포함)
├── datasets.py          # 데이터셋/쿼리 설정 (공유)
├── rlm_optimized.py     # 최적화 RLM (캐싱, 병렬)
├── benchmark.py         # 벤치마크 유틸리티 (라이브러리)
├── extract_documents.py # PDF/DOCX 텍스트 추출
├── rlm/
│   ├── baseline.py      # 원 논문 방식
│   ├── base.py          # 공통 베이스 클래스
│   └── config.py        # 가격 설정
├── repl_optimized.py    # 최적화 REPL
└── llm_client.py        # Gemini API 클라이언트
```

## 데이터셋

| ID | 이름 | 설명 |
|----|------|------|
| 1 | NSMC | 네이버 영화 리뷰 (자동 다운로드) |
| 2 | Wiki | 한국어 위키피디아 샘플 |
| 3 | Law Insider | 법률 계약서 (extract_documents.py 실행 필요) |

## 개선 사항 (vs 원 논문)

| 기능 | Baseline | Optimized |
|------|----------|-----------|
| llm_query | 순차 | 캐싱 |
| llm_query_batch | ❌ | ✅ 병렬 |
| RecursionGuard | ❌ | ✅ |
| 비용 추적 | ❌ | ✅ |
| 출력 Truncation | ❌ | ✅ |

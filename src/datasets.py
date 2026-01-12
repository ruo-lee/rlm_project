"""
Shared configuration for datasets and test queries.
Used by both main.py and benchmark.py.
"""

# ============================================================================
# DATASETS - Available data sources for RLM
# ============================================================================
DATASETS = {
    "1": {
        "name": "NSMC (네이버 영화 리뷰)",
        "path": "data/ratings_train.txt",
        "url": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    },
    "2": {
        "name": "Korean CSV/Text (Wiki Sample)",
        "path": "data/wiki_ko_sample.txt",
        "url": None,  # Local only
    },
    "3": {
        "name": "Law Insider (법률 계약서)",
        "path": "data/law_insider_combined.txt",
        "url": None,  # Local only - run extract_documents.py first
    },
}

# ============================================================================
# TEST QUERIES - Predefined queries for testing
# ============================================================================
TEST_QUERIES = {
    # NSMC Queries
    "1": {
        "name": "긍정 단어 분석 (Simple)",
        "query": "이 데이터셋에서 가장 많이 등장하는 긍정적인 단어 3개를 찾아줘. 그리고 2023년이라는 숫자가 포함된 리뷰가 있는지 확인해줘.",
        "description": "단순 집계 작업 - llm_query_batch 사용 예상",
    },
    "2": {
        "name": "감정 분포 분석 (Medium)",
        "query": "긍정(label=1)과 부정(label=0) 리뷰의 평균 길이를 비교하고, 각각에서 가장 자주 사용되는 감정 표현 패턴을 분석해줘.",
        "description": "비교 분석 - 약간의 복잡도",
    },
    "3": {
        "name": "섹션별 요약 (Complex - RLM 재귀 권장)",
        "query": "데이터를 1000개씩 5개 섹션으로 나누고, 각 섹션별로 '주요 감정 키워드'와 '대표 리뷰'를 요약해줘. 그리고 전체적인 트렌드를 종합해줘.",
        "description": "복잡한 다단계 작업 - RLM() 재귀 호출 권장",
    },
    "4": {
        "name": "비교 분석 (Complex - RLM 재귀 권장)",
        "query": "긍정 리뷰 500개와 부정 리뷰 500개를 각각 분석해서, 긍정에서만 나타나는 단어와 부정에서만 나타나는 단어를 찾고, 그 차이를 설명해줘.",
        "description": "비교 대조 분석 - RLM() 재귀 호출 권장",
    },
    # Wiki Queries
    "5": {
        "name": "[Wiki] 조선시대 문서 찾기 (Verifiable)",
        "query": "이 데이터셋에서 '조선'이라는 단어가 포함된 문서는 몇 개인지 세어주고, 그 중 가장 긴 문서의 제목을 알려줘.",
        "description": "검증 가능한 검색 작업 (grep/wc로 확인 가능)",
    },
    "6": {
        "name": "[Wiki] 주제 분류 (Complex)",
        "query": "전체 문서를 훑어보고 주요 카테고리 5개를 정의한 뒤, 각 카테고리에 속하는 대표적인 문서 제목을 3개씩 나열해줘.",
        "description": "전체적인 이해와 분류 능력 테스트",
    },
    # Custom
    "7": {
        "name": "Custom Query",
        "query": None,
        "description": "직접 질문 입력",
    },
    # Law Insider Queries
    "8": {
        "name": "[Law] 계약 조건 분석 (Simple)",
        "query": "이 계약서들에서 'borrower'와 'lender'의 주요 의무사항을 요약해줘.",
        "description": "계약 당사자 의무 분석",
    },
    "9": {
        "name": "[Law] 정의 조항 찾기 (Verifiable)",
        "query": "각 계약서의 'Definitions' 섹션에서 정의된 주요 용어 5개를 나열해줘.",
        "description": "법률 문서 정의 조항 추출",
    },
    "10": {
        "name": "[Law] 계약서 비교 (Complex - RLM 재귀 권장)",
        "query": "각 계약서별로 대출 조건(금액, 이자율), 상환 조건, 위약 조항을 분석하고, 공통점과 차이점을 정리해줘.",
        "description": "다수 법률 문서 비교 분석 - RLM() 재귀 호출 권장",
    },
}

# ============================================================================
# CONTEXT SIZES - Available context size options
# ============================================================================
CONTEXT_SIZES = {
    "100k": 100000,
    "500k": 500000,
    "1m": 1000000,
    "full": None,  # Will be set to full length at runtime
}

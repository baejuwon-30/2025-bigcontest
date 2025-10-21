# Store Agents Pipeline (3 Agents, Universal via JSON->Gemini)

## Agents
1) 🧭 우리 가게 데이터 분석 — Milvus 컬렉션 `store_kpi_collection` 사용
2) 🌐 보편성 분석 — **매장명 입력 → data/{매장명}_persona_report.json 로드 → Gemini가 모든 계산/정렬/매핑 수행**
3) 🏆 1위 가게 비교 — Milvus 컬렉션 `leader_benchmark_collection` 사용

**샘플 데이터**: `data/난포_persona_report.json` (업로드되어 있으면 포함됨)

## Setup
```bash
pip install -r requirements.txt
# .streamlit/secrets.example.toml -> .streamlit/secrets.toml 로 복사 후 값 채우기
```

## Run
```bash
streamlit run app.py
```

## Notes
- Milvus 컬렉션명/필드명은 실제 스키마에 맞게 `app.py` 상단에서 수정하세요.
- 보편성(2번) 에이전트는 RAG 불필요. JSON 하나로 완결됩니다.

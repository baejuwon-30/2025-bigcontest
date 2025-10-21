# app.py — Store Agents (3 buttons) + Universal(JSON→Gemini) + VS Leader(Prebuilt JSON explain)
# - 버튼 3개: 우리 가게 데이터 분석(1) / 온라인 대중성 스코어 분석(2) / 1위 가게 vs 우리 가게 분석(3)
# - (1) Store Analysis: Milvus(HF 임베딩) RAG로 상위 문맥 수집 → Gemini 보고서 생성(마크다운)
# - (2) Universal: data/{매장명}_persona_report.json → Gemini가 지표/정렬/액션 매핑
# - (3) VS Leader: 사전 생성된 data/{우리}_vs_leader.json을 로드해 설명(LLM 호출 없음)

import json
from pathlib import Path
import streamlit as st
from jinja2 import Template
import google.generativeai as genai  # 1,2번에서 LLM 호출
from pymilvus import MilvusClient

# ================== Page ==================
st.set_page_config(page_title="세 가지 에이전트로 가게 분석하기", page_icon="🧭", layout="centered")
st.title("세 가지 에이전트로 우리 가게 분석하기")

# ================== Secrets ==================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
ZILLIZ_URI     = st.secrets.get("ZILLIZ_URI")
ZILLIZ_TOKEN   = st.secrets.get("ZILLIZ_TOKEN")
HF_TOKEN       = st.secrets.get("HF_TOKEN")
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # Gemini 사용하므로 불필요

# ================== Cached clients ==================
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml을 확인하세요.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

@st.cache_resource(show_spinner=False)
def get_milvus():
    if not (ZILLIZ_URI and ZILLIZ_TOKEN):
        return None
    return MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    if not GEMINI_API_KEY:
        return None
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

# ================== Constants (Milvus 스키마 맞춤) ==================
STORE_ANALYSIS_COLLECTION = "shinahn_collection_hf"       # 업로드 코드와 동일
OUTPUT_FIELDS             = ["text", "description"]       # 스키마에 맞춤
# (참고) VS_LEADER_COLLECTION 은 현재 사용 안 함 (3번은 사전 JSON 렌더만)

# ================== Utilities ==================
def embed_query(query: str):
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("HF_TOKEN이 설정되지 않아 임베딩 모델을 만들 수 없습니다.")
    return model.embed_query(query)

def milvus_search_topk(query_vec, top_k=3, output_fields=None):
    milvus = get_milvus()
    if milvus is None:
        raise RuntimeError("Milvus 클라이언트가 설정되지 않았습니다. ZILLIZ_URI/ZILLIZ_TOKEN을 secrets에 넣으세요.")
    output_fields = output_fields or OUTPUT_FIELDS
    res = milvus.search(
        collection_name=STORE_ANALYSIS_COLLECTION,
        data=[query_vec],
        limit=top_k,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE"}
    )
    hits = []
    raw = res[0] if res else []
    for hit in raw:
        # pymilvus MilvusClient의 결과 dict 기반
        if isinstance(hit, dict):
            ent = hit.get("entity") or {}
            row = {f: ent.get(f) for f in output_fields}
        else:
            row = {}
            for f in output_fields:
                val = getattr(hit, f, None)
                if val is None and hasattr(hit, "entity"):
                    try:
                        val = hit.entity.get(f)
                    except Exception:
                        val = None
                row[f] = val
        hits.append(row)
    return hits

def build_context_text(hits, max_chars=4000):
    """text와 description을 묶어 컨텍스트로 생성"""
    pieces = []
    for i, h in enumerate(hits, 1):
        content = (h.get("text") or "").strip()
        desc    = (h.get("description") or "").strip()
        if not content and not desc:
            continue
        blob = f"[{i}] {content}"
        if desc:
            blob += f"\n(설명: {desc})"
        pieces.append(blob)
    ctx = "\n\n".join(pieces)
    return ctx[:max_chars]

def call_gemini_text(prompt_text: str, model: str = "gemini-2.5-flash") -> str:
    client = get_gemini_client()
    model_instance = client.GenerativeModel(model)
    resp = model_instance.generate_content(prompt_text)
    return (resp.text or "").strip()

def call_gemini_json(prompt_text: str, model: str = "gemini-2.5-flash"):
    """JSON만 뽑아야 하는 2번/기타 용도"""
    txt = call_gemini_text(prompt_text, model=model)
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("Gemini가 JSON을 반환하지 않았습니다.")
    return json.loads(txt[s:e+1])

# ================== (1) Store Analysis: RAG 프롬프트 ==================
STORE_RAG_PROMPT = """
당신은 한국어로 쓰는 리테일 데이터 애널리스트입니다.
아래는 특정 가맹점 관련 RAG 컨텍스트입니다(문서 본문 + 설명 메타). 이 정보를 **사실 위주로** 요약하고 실행 조언을 제시하세요.

도메인 규칙(컨텍스트에 포함된 지표 해석):
- 값이 -999999.99 인 경우: **결측**(정보 없음)
- '가맹점 운영 개월수 구간': **0%에 가까울수록 상위**(운영 개월 수가 높음)
- '매출금액 구간': **0%에 가까울수록 상위**(매출 금액이 높음)
- '매출건수 구간': **0%에 가까울수록 상위**(거래 건수가 많음)
- '유니크 고객 수 구간': **0%에 가까울수록 상위**(고객 풀이 큼)
- 날짜/월별 지표는 **최근 24개월의 추세**가 중요. 이상치는 -999999.99로 표기될 수 있음.

출력 형식(마크다운, 한국어):
# 가맹점 스냅샷
- 핵심 요약 3줄(매출/방문/고객풀 관점)
- 최근 추세(증가/정체/감소) 한 줄

# 강점과 리스크
- **강점(3~5)**: 근거를 짧게
- **리스크(3 이하)**: 근거/영향을 짧게

# 실행 제안(우선순위 순, 3~5개)
- {구체적 액션} — {기대효과/측정지표}

# 참고 컨텍스트(요약)
- [1]~[3]에서 근거가 된 문구를 **한 줄씩** 요약

금지:
- 컨텍스트에 없는 사실/수치 **추가로 만들지 말 것**
- 지나치게 일반적인 조언 금지(이 가맹점의 맥락 반영)
- 내부 지표 정의를 바꾸지 말 것

[컨텍스트 시작]
{context}
[컨텍스트 끝]

사용자 질문: "{question}"
""".strip()

def run_agent_store_analysis_report(user_query: str):
    """질문(가맹점명/분석요청)을 임베딩→Milvus검색→컨텍스트→Gemini 보고서로 생성"""
    vec  = embed_query(user_query)
    hits = milvus_search_topk(vec, top_k=3, output_fields=OUTPUT_FIELDS)
    ctx  = build_context_text(hits)
    prompt = STORE_RAG_PROMPT.format(context=ctx, question=user_query)
    report_md = call_gemini_text(prompt)
    return report_md, hits


# ================== Agent 2: 보편성 분석 (JSON -> Gemini 계산/정렬/매핑) ==================
PERSONA_AGENT_PROMPT = """

"""

def run_agent_universal(store_name: str):
    json_path = Path("data") / f"{store_name}_persona_report.json"
    if not json_path.exists():
        raise FileNotFoundError(f"리포트 파일이 없습니다: {json_path}")
    raw = json_path.read_text(encoding="utf-8")
    prompt = PERSONA_AGENT_PROMPT + raw + "\n=== JSON REPORT END ==="
    return call_gemini_json(prompt)

def render_persona_dashboard(result: dict):
    st.subheader("고객 매력도 종합 리포트")
    m = result["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("우리 가게가 받는 종합 호감도", f"{m['appeal']:.2f}")
    c2.metric("페르소나별 호감도 균형지수", f"{m['balance_cv']:.4f}")
    c3.metric("퍼소나 긍정 반응률", f"{m['coverage']*100:.1f}%")

    st.subheader("페르소나 순위 (total_sum 내림차순)")
    for r in result["personas_sorted"]:
        header = f"{r['label']} — total {r['total_sum']}  (per review {r['per_review']:.3f}, reviews {r['reviews']})"
        with st.expander(header):
            st.write("**Pros**")
            for p in r.get("pros", []): st.write("- ", p)
            st.write("**Cons**")
            for c in r.get("cons", []): st.write("- ", c)
            st.write("**Suggestions**")
            for s in r.get("suggestions", []): st.write("- ", s)

    st.subheader("문제 ↔ 해결 전략 1:1 매핑")
    for pair in result.get("action_map", []):
        st.markdown(f"⚠️ **문제:** {pair['con']}  \n☑️ **조치:** {pair['suggestion']}")

# ================== (참고) Agent 3 LLM 프롬프트 (현재 사용 안 함: 사전 생성 JSON만 렌더)
VS_LEADER_PAIRWISE_JSON_PROMPT = """
(생략)  # 필요 시 다시 활성화
"""

# ================== VS Leader 비교 JSON 렌더 (옵션 B: 구조화 출력) ==================
def render_vs_leader_pack(pack: dict) -> None:
    """사전 생성된 비교 JSON(dict)을 읽어 화면에 설명 (옵션 B: 내러티브에서 [개선 제안] 제외, actions를 별도로 1회 표시)."""
    store_our = pack.get("store_name_our", "우리 가게")
    store_leader = pack.get("store_name_leader", "1위 가게")
    st.subheader(f"비교 결과: {store_our} vs {store_leader}")


    st.subheader("페르소나별 격차 분석")
    personas = pack.get("personas", [])
    if not personas:
        st.info("페르소나 항목이 없습니다.")
        return

    for p in personas:
        label = p.get("label", p.get("persona_id", "알 수 없음"))
        st.markdown(f"**{label}**")

        # 점수 요약
        s_our = p.get("score_ours")
        s_lead = p.get("score_leader")
        gap = p.get("gap")
        if s_our is not None and s_lead is not None and gap is not None:
            st.caption(f"점수: 우리 {s_our} / 1위 {s_lead} (gap {gap:+})")

        # 내러티브: [개선 제안] 섹션 이전까지만 출력
        narrative = (p.get("narrative") or "").strip()
        if narrative:
            text_no_actions = narrative.split("[개선 제안]")[0].strip()
            if text_no_actions:
                st.write(text_no_actions)

        # 액션: 구조화 배열을 사용하여 한 번만 표시 + 중복 제거
        actions_in = p.get("actions") or []
        dedup_actions = []
        seen = set()
        for a in actions_in:
            s = (a or "").strip()
            if s and s not in seen:
                seen.add(s)
                dedup_actions.append(s)
        if dedup_actions:
            st.markdown("**[개선 제안]**")
            for a in dedup_actions:
                st.write("- ", a)

    roadmap = pack.get("roadmap") or []
    if roadmap:
        st.subheader("우선 실행 로드맵")
        for line in roadmap:
            st.write(" ", line)

# ================== Agent 3 (Milvus 템플릿 버전이 필요하면 사용) ==================
def run_agent_vs_leader_rag(user_query: str):
    vec  = embed_query(user_query)
    hits = milvus_search(VS_LEADER_COLLECTION, vec, top_k=5)
    ctx  = build_context_text(hits)
    tmpl = load_prompt_template("vs_leader_ko")  # prompts/vs_leader_ko.jinja 필요
    prompt = tmpl.render(question=user_query, context=ctx)
    out = call_gemini_json(prompt)
    return out, hits

# ================== Agent selector (3 Buttons) ==================
if "agent" not in st.session_state:
    st.session_state.agent = None

AGENTS = {
    "store_analysis": {
        "name": "🔎 우리 가게 데이터 분석",
        "desc": "내부 KPI·고객행동을 요약하고 병목을 찾아 개선안을 제시합니다."
    },
    "universal": {
        "name": "🌐 온라인 대중성 스코어 분석",
        "desc": "Naver 지도 리뷰 데이터를 기반으로, 가게의 '온라인 대중성'과 '고객층별 매력도'를 종합적으로 평가합니다."
    },
    "vs_leader": {
        "name": "🏆 1위 가게 vs 우리 가게 분석",
        "desc": "사전 생성된 비교 JSON을 불러와 페르소나별 점수·갭·개선안을 설명합니다."
    },
}

st.markdown("#### 에이전트 선택")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button(AGENTS["store_analysis"]["name"]):
        st.session_state.agent = "store_analysis"
with c2:
    if st.button(AGENTS["universal"]["name"]):
        st.session_state.agent = "universal"
with c3:
    if st.button(AGENTS["vs_leader"]["name"]):
        st.session_state.agent = "vs_leader"

agent = st.session_state.get("agent")
if not agent:
    st.info("상단 버튼 중 하나를 눌러 에이전트를 선택하세요.")
    st.stop()
else:
    st.success(AGENTS[agent].get("desc", AGENTS[agent]["name"]))

# ================== Branch: 2) Universal (JSON -> Gemini) ==================
if agent == "universal":
    store_in = st.chat_input("가게명을 입력하세요 (예: 데판야끼현, 명가떡볶이, 크레송, 타코마이너 등)")
    if store_in:
        with st.spinner("우리 가게의 온라인 대중성 스코어 분석 중..."):
            try:
                result = run_agent_universal(store_in.strip())
                render_persona_dashboard(result)
            except Exception as e:
                st.error(f"오류: {e}")

# ================== Branch: 3) VS Leader (사전 생성된 비교 JSON 자동 로드) ==================
elif agent == "vs_leader":
    our_store = st.text_input("우리 가게명", placeholder="예: 데판야끼현 / 명가떡볶이 / 크레송 / 타코마이너")

    if st.button("비교 결과 불러오기", type="primary", use_container_width=True):
        try:
            if not our_store or not our_store.strip():
                raise ValueError("우리 가게명을 입력하세요.")
            path = Path("data") / f"{our_store.strip()}_vs_leader.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"비교 데이터 파일을 찾을 수 없습니다: {path}\n"
                    "→ 파일을 준비해 주세요. 예) data/난포_vs_leader.json"
                )
            pack = json.loads(path.read_text(encoding="utf-8"))
            render_vs_leader_pack(pack)

        except Exception as e:
            st.error(f"오류: {e}")

# ================== Branch: 1) Store Analysis (Milvus RAG) ==================
elif agent == "store_analysis":
    q = st.chat_input("질문을 입력하세요 (예: 우리 가게 점심 전환율?)")
    if q:
        with st.spinner("분석 중..."):
            try:
                report_md, hits = run_agent_store_analysis_report(q)
                st.markdown(report_md)
                with st.expander("참고 문맥(검색 상위 결과)"):
                    for i, h in enumerate(hits, 1):
                        st.markdown(f"**[{i}]** {h.get('content','')}")
                        if h.get("source"):
                            st.caption(f"출처: {h['source']}")
            except Exception as e:
                st.error(f"오류: {e}")

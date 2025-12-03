import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 기본 설정 ---
st.set_page_config(page_title="유튜브 금융 인사이트 DB", layout="wide")

# Gemini API 설정
if "google_api_key" in st.secrets:
    genai.configure(api_key=st.secrets["google_api_key"])
else:
    st.error("🚨 Google API 키가 설정되지 않았습니다. Secrets를 확인해주세요.")

# --- 2. 구글 시트 연결 함수 ---
@st.cache_resource
def init_connection():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

def get_data():
    client = init_connection()
    sheet = client.open("Youtube_Data_Store").sheet1 
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(row_data):
    client = init_connection()
    sheet = client.open("Youtube_Data_Store").sheet1
    sheet.append_row(row_data)
    # 데이터가 갱신되었으므로 캐시 삭제 (챗봇이 새 데이터를 알게 함)
    st.cache_resource.clear()

# --- 3. 검색 엔진 (TF-IDF) ---
def search_documents(query, df, top_k=3):
    if df.empty:
        return []
    
    # 검색 정확도를 높이기 위해 '주장'과 '시사점'까지 검색 범위에 포함
    df['combined_text'] = (
        df['제목'].astype(str) + " " + 
        df['핵심주제'].astype(str) + " " + 
        df['핵심주장'].astype(str) + " " + 
        df['시사점'].astype(str) + " " + 
        df['요약'].astype(str)
    )
    
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        query_vec = tfidf.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if cosine_sim[idx] > 0:
                results.append(df.iloc[idx])
        return results
    except ValueError:
        return []

# --- 4. 메인 UI ---
st.title("📺 유튜브 금융 인사이트 저장소 (Full Ver.)")

tab1, tab2 = st.tabs(["📥 데이터 입력", "🤖 AI 챗봇"])

# === [탭 1] 데이터 입력 ===
with tab1:
    st.subheader("Gemini 분석 데이터 적재")
    st.info("💡 JSON의 모든 정보(주장, 근거, 시사점 등)를 빠짐없이 저장합니다.")
    
    with st.form("data_input_form"):
        json_input = st.text_area("JSON Input", height=200, placeholder="Gemini가 준 JSON 코드를 붙여넣으세요.")
        submitted = st.form_submit_button("DB 저장하기")

    if submitted and json_input:
        try:
            data = json.loads(json_input)
            
            # 리스트 형태의 데이터(주장, 근거)를 줄바꿈 문자열로 변환
            key_arguments = "\n- ".join(data.get("key_arguments", []))
            if key_arguments: key_arguments = "- " + key_arguments
            
            evidence = "\n- ".join(data.get("evidence", []))
            if evidence: evidence = "- " + evidence

            # 구글 시트 컬럼 순서에 맞춰 데이터 준비 (14개 항목)
            row_data = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # A: 수집일시
                data.get("published_at", ""),                 # B: 업로드일
                data.get("video_id", ""),                     # C: 영상ID
                data.get("title", ""),                        # D: 제목
                data.get("channel_name", ""),                 # E: 채널명
                data.get("main_topic", ""),                   # F: 핵심주제
                key_arguments,                                # G: 핵심주장 (상세)
                evidence,                                     # H: 근거 (상세)
                data.get("implications", ""),                 # I: 시사점
                data.get("validity_check", ""),               # J: 타당성
                data.get("sentiment", ""),                    # K: 감정
                data.get("full_summary", ""),                 # L: 요약
                data.get("tags", ""),                         # M: 태그
                data.get("url", "")                           # N: URL
            ]
            
            save_data(row_data)
            st.success(f"✅ 모든 상세 정보 저장 완료: {data.get('title')}")
            
        except json.JSONDecodeError:
            st.error("❌ JSON 형식이 올바르지 않습니다.")
        except Exception as e:
            st.error(f"❌ 저장 중 오류 발생: {e}")

# === [탭 2] AI 챗봇 ===
with tab2:
    st.subheader("내 금융 데이터와 대화하기")
    
    try:
        df = get_data()
        st.caption(f"📚 현재 총 {len(df)}개의 심층 분석 데이터가 학습되어 있습니다.")
    except:
        st.warning("데이터를 불러올 수 없습니다. 구글 시트 헤더(1행)를 확인해주세요.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 저장된 심층 분석 데이터를 바탕으로 답변해 드릴게요."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Gemini가 인사이트를 분석 중입니다..."):
            relevant_rows = search_documents(prompt, df)
            
            if not relevant_rows:
                response_text = "죄송합니다. 관련된 내용을 DB에서 찾을 수 없습니다."
            else:
                context_str = ""
                for idx, row in enumerate(relevant_rows):
                    # 챗봇에게 더 풍부한 정보를 줍니다
                    context_str += f"""
                    [참고 영상 {idx+1}]
                    - 제목: {row['제목']} (채널: {row['채널명']})
                    - 핵심주장: {row['핵심주장']}
                    - 시사점: {row['시사점']}
                    - 타당성 평가: {row['타당성']}
                    - 요약: {row['요약']}
                    """
                
                system_prompt = f"""
                당신은 전문 투자 자문 AI입니다. 아래 [참고 영상]의 심층 분석 데이터를 바탕으로 질문에 답변하세요.
                단순한 요약보다는 '핵심 주장', '시사점', '근거'를 중심으로 논리적인 답변을 하세요.
                
                [참고 영상 데이터]
                {context_str}
                
                [질문]
                {prompt}
                """
                
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(system_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"AI 답변 생성 실패: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)

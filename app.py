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
    st.cache_resource.clear()

# --- 3. 검색 엔진 (TF-IDF) ---
def search_documents(query, df, top_k=3):
    if df.empty:
        return []
        
    df['combined_text'] = df['title'].astype(str) + " " + df['main_topic'].astype(str) + " " + df['full_summary'].astype(str)
    
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
st.title("📺 유튜브 금융 인사이트 저장소 (Powered by Gemini)")

tab1, tab2 = st.tabs(["📥 데이터 입력", "🤖 AI 챗봇"])

# === [탭 1] 데이터 입력 ===
with tab1:
    st.subheader("Gemini 분석 데이터 적재")
    with st.form("data_input_form"):
        json_input = st.text_area("JSON Input", height=200, placeholder="Gemini가 준 JSON 코드를 붙여넣으세요.")
        submitted = st.form_submit_button("DB 저장하기")

    if submitted and json_input:
        try:
            data = json.loads(json_input)
            row_data = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("video_id", ""),
                data.get("title", ""),
                data.get("channel_name", ""),
                data.get("main_topic", ""),
                data.get("full_summary", ""),
                data.get("tags", ""),
                data.get("url", "")
            ]
            save_data(row_data)
            st.success(f"✅ 저장 완료: {data.get('title')}")
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

# === [탭 2] AI 챗봇 ===
with tab2:
    st.subheader("내 금융 데이터와 대화하기")
    
    try:
        df = get_data()
        st.caption(f"📚 현재 총 {len(df)}개의 영상 데이터가 학습되어 있습니다.")
    except:
        st.warning("아직 데이터가 없거나 구글 시트 연결에 실패했습니다.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! Gemini입니다. 저장된 데이터를 바탕으로 답변해 드릴게요."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Gemini가 데이터를 분석 중입니다..."):
            # 1. 관련 데이터 검색
            relevant_rows = search_documents(prompt, df)
            
            # 2. 프롬프트 구성
            if not relevant_rows:
                response_text = "죄송합니다. 관련된 내용을 DB에서 찾을 수 없습니다."
            else:
                context_str = ""
                for idx, row in enumerate(relevant_rows):
                    context_str += f"\n[참고 영상 {idx+1}]\n- 제목: {row['title']}\n- 채널: {row['channel_name']}\n- 내용: {row['full_summary']}\n"
                
                system_prompt = f"""
                당신은 금융 투자 어시스턴트입니다. 다음 [참고 영상] 데이터를 바탕으로 질문에 답변하세요.
                
                [참고 영상]
                {context_str}
                
                [질문]
                {prompt}
                
                답변 시 출처(영상 제목, 채널)를 명시해주세요.
                """
                
                # 3. Gemini 호출
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(system_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"AI 오류 발생: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)

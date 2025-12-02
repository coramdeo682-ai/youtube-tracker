import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="유튜브 데이터 수집기", layout="wide")

st.title("📺 유튜브 데이터 DB 적재 (Mobile Ver.)")

# 1. 구글 시트 연결 함수
def get_google_sheet():
    # Streamlit Cloud의 Secrets에서 인증 정보 가져오기
    # (주의: 로컬에서 테스트할 때와 클라우드 배포 시 설정이 다릅니다. 아래는 클라우드용)
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    # st.secrets에 저장된 JSON 정보를 사용
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    # 구글 시트 이름으로 열기 (반드시 공유가 되어 있어야 함)
    # 시트 이름이 'Youtube_Data_Store'라고 가정
    sheet = client.open("Youtube_Data_Store").sheet1 
    return sheet

# 2. 입력 폼
with st.form("data_input_form"):
    st.info("Gemini가 생성한 JSON 코드를 아래에 붙여넣으세요.")
    json_input = st.text_area("JSON Input", height=300)
    submitted = st.form_submit_button("DB 저장하기")

# 3. 저장 로직
if submitted and json_input:
    try:
        # JSON 문자열을 파이썬 딕셔너리로 변환
        data = json.loads(json_input)
        
        # 필요한 데이터 추출 (리스트 형태로 변환)
        # 구글 시트의 컬럼 순서: [날짜, 영상ID, 제목, 채널명, 핵심주제, 요약, 태그, URL]
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # 수집 일시
            data.get("video_id", ""),
            data.get("title", ""),
            data.get("channel_name", ""),
            data.get("main_topic", ""),
            data.get("full_summary", ""),
            data.get("tags", ""),
            data.get("url", "")
        ]
        
        # 구글 시트에 추가
        sheet = get_google_sheet()
        sheet.append_row(row_data)
        
        st.success(f"✅ 저장 완료! : {data.get('title')}")
        
    except json.JSONDecodeError:
        st.error("❌ JSON 형식이 올바르지 않습니다. 코드를 다시 확인해주세요.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "프로젝트 주제가 뭔가요?",
    "모델은 어떤걸 썼나요?",
    "그 모델을 선택한 이유가 뭔가요?",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트의 목표는 무엇인가요?",
    "프로젝트 기간은 어떻게 되나요?",
    "데이터는 뭘 이용하였나요?"
]

answers = [
    "내시경(대/위장) 실시간 진단입니다.",
    "YOLO8 seg 버전을 썼습니다.",
    "의료 이미지에서 병변 영역을 정확히 감지하고 분할(Segmentation)하기 위해서 해당 모델을 선택했습니다.",
    "3명입니다.",
    "내시경 시 놓칠 수 있는 병변을 실시간으로 발견하는 것이 목표입니다.",
    "3주입니다.",
    "ai-hub 데이터와 실시간 내시경 유튜브 영상을 활용하였습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("혜정's 프로젝트 챗봇")
st.write("프로젝트에 관한 질문을 입력해보세요. 예: 프로젝트 주제가 뭔가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")

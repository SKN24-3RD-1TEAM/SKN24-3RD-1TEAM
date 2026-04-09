from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
import streamlit as st
import chromadb
import base64

from src.models import load_models, load_embeddings
from src.graph import build_graph
from src.utils.get_persona import generate_king_config
from src.utils.gen_scene import generate_initial_scene

st.markdown("""
<style>
#MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
.stApp { background-color: #F8EEE6; }
</style>
""", unsafe_allow_html=True)

# 페이지 세션 관리
if 'page' not in st.session_state:
    st.session_state.page = 'main'          # 메인으로 고정
    st.session_state.app_graph = None
    st.session_state.llm_generator = None
    st.session_state.king_data = None
    st.session_state.user_role = None
    st.session_state.messages = []
    st.session_state.anger_level = 0
    st.session_state.scenario = ""

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

st.set_page_config(layout="wide")

st.markdown("""
<style>
@font-face { font-family: 'HsBombaram30'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_20-10@1.1/HSBombaram3_Thin.woff') format('woff'); }
@font-face {
    font-family: 'Shilla';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2206-02@1.0/Shilla_CultureM-Medium.woff2') format('woff2');
    font-weight: 500;
    font-display: swap;
}
</style>
""", unsafe_allow_html=True)

# 메인 페이지
if st.session_state.page == 'main':

    img_base64 = image_to_base64("./data/images/king.png")
    st.markdown(f"""
        <div style="display:flex; justify-content:center; margin-left:200px; margin-top:-100px;">
            <img src="data:image/png;base64,{img_base64}" width="600"/>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @font-face { font-family: 'HsBombaram30'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_20-10@1.1/HSBombaram3_Thin.woff') format('woff'); }
    .king_select2 p {
        margin-top: -340px;
        font-family: 'Shilla';
        font-size: 150px !important;

    background: linear-gradient(to right, 
            #EFC554 0%, 
            #FDC13A 20%, 
            #F8E47B 50%, 
            #EAD779 80%, 
            #FDC13A 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(5px 5px 10px rgba(0, 0, 0, 0.5));
        letter-spacing: 5px;
    }
    .king_select p { margin-top: -10px; color: #000; font-family: 'Shilla'; font-size: 40px !important; }
    div[data-testid="stSelectbox"] {
        margin-top: -10px; 
    }
    </style>
    <div class='king_select2' style="display:flex; justify-content:center;"><p>조선킹덤</p></div>
    <div class='king_select' style="display:flex; justify-content:center;"><p>왕을 선택해주세요</p></div>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([2, 1, 2])
    with col_mid:
        st.markdown("<p style='text-align: center; color: #F8EEE6; font-size: 24px; font-weight: bold; margin-bottom: -10px;'>과인께 아뢸 말씀을 고르거라</p>", unsafe_allow_html=True)
        option = st.selectbox("label_hidden", ["태조", "세종", "단종", "세조", "연산군"], label_visibility="collapsed")
        st.markdown(
                """
                    <style>
                    .stButton > button {
                        margin-top: 10px;
                        margin-left: 70px;

                        border: 2px solid #D4AF37;
                        border-radius: 8px;

                        background-color: #000;
                        color: #D4AF37;
                        font-weight: bold;
                        font-size: 52px;

                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        transition: all 0.3s ease;

                        height: 60px;
                    }
                    .stButton > button p {
                        font-size: 24px;
                        font-family: 'Shilla';
                    }

                    .stButton > button:hover {

                        border: 2px solid #FFD700 !important;
                        color: #FFD700 !important;
                        background-color: #fff !important;

                        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5) !important;
                        transform: translateY(-3px) !important;
                    }


                    .stButton > button:active {
                        transform: translateY(0);
                    }
                    /* 스피너(로더) 색상을 블랙으로 변경 */
                    div.stSpinner  {
                        color: #000 !important;
                        border-top-color: #000 !important;
                    }
                    div.stSpinner span {
                        color: #000 !important;
                        background-color: #000 !important;
                    }

                    /* 스피너 아래 텍스트 컬러도 블랙으로 */
                    div[data-testid="stStatusWidget"] {
                        color: #000 !important;
                        background-color: #000 !important;
                        font-size: 20px;
                    }

                    div[data-testid="stLoader"] {
                    /* 배경이 되는 연한 선 제거 또는 블랙 투명하게 */
                        border: 3px solid rgba(0, 0, 0, 0.1) !important;
                        /* 실제로 돌아가는 진한 선 블랙으로 */
                        border-top: 3px solid #000 !important;
                    }

                    /* 스피너와 함께 뜨는 상태 텍스트 블랙 처리 */
                    div[data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {
                        color: #000 !important;
                        font-weight: bold;
                    }
                    </style>
                    """,
            unsafe_allow_html=True

            )
        if st.button("대화 시작"):
            with st.spinner("로컬 모델 및 벡터 DB를 불러오는 중입니다..."):
                llms = load_models()
                st.session_state.llm_generator = llms["generator"]
                
                embeddings = load_embeddings()
                persistent_client = chromadb.PersistentClient(path="./database/reference_db")
                vectorstore = Chroma(client=persistent_client, embedding_function=embeddings, collection_name='reference')

                st.session_state.app_graph = build_graph(llms["analyzer"], llms["generator"], vectorstore)
                
                king_name = '태조 이성계' if option == '태조' else option
                st.session_state.king_name = king_name
                st.session_state.king_data = generate_king_config(king_name, llm=st.session_state.llm_generator)
                
                # 다음 페이지로 전환
                st.session_state.page = 'role_select'
                st.rerun()


# 신분 선택 페이지
elif st.session_state.page == 'role_select':
    st.markdown(f"<h1 style='text-align: center; color: #000; font-family: Shilla; font-size:55px;'>{st.session_state.king_name} 전하를 뵙기 전</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #000; font-family: Shilla;'>본인의 신분을 선택해주세요</h2>", unsafe_allow_html=True)
    st.markdown("""
            <style>
                p{
                    color: #000;
                    font-size: 24px !important;
                    font-family: 'Shilla'
                }
                .stButton > button p {
                        font-size: 24px;
                        font-family: 'Shilla';
                color: #fff;
                }
                #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-zy6yx3.e1td4qo64 > div > div:nth-child(6) > div > label > span > div > p{
                font-size: 50px !important;
                color: #D4AF37;
                }
                /* 라디오 버튼의 전체 컨테이너 정렬 */
            div[data-testid="stWidgetLabel"] {
                display: flex;
                align-items: center; /* 세로 기준 중앙 */
                justify-content: center; /* 가로 기준 중앙 */
            }

            /* 라디오 버튼 각각의 항목(동그라미 + 글자) 정렬 */
            div[data-testid="stHorizontalBlock"] label {
                display: flex;
                align-items: center !important;
                justify-content: center;
                height: 100%; /* 부모 높이에 맞춤 */
            }

            /* 라디오 버튼 동그라미 자체 위치 미세 조정 (필요시) */
            div[data-testid="stMarkdownContainer"] p {
                margin-bottom: 0px !important;
            }
                
                <style>
                    .stButton > button {
                        margin-top: 10px;
                        margin-left: 70px;

                        border: 2px solid #D4AF37;
                        border-radius: 8px;

                        background-color: #000;
                        color: #D4AF37;
                        font-weight: bold;
                        font-size: 52px;

                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        transition: all 0.3s ease;

                        height: 60px;
                    }
                    .stButton > button p {
                        font-size: 24px;
                        font-family: 'Shilla';
                    }

                    .stButton > button:hover {

                        border: 2px solid #FFD700 !important;
                        color: #000 !important;
                        background-color: #000 !important;

                        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5) !important;
                        transform: translateY(-3px) !important;
                    }


                    .stButton > button:active {
                        transform: translateY(0);
                    }

            </style>
            """, unsafe_allow_html=True)
    roles = ["백성", "유생", "신하", "장수", "상인", "의원", "내관/궁인", "중국 황제", "학생"]
    role_descriptions = {
        "백성": "왕과 가장 큰 거리가 있는 존재. 극도의 긴장과 두려움.",
        "유생": "예를 갖추어 아뢰지만, 학문과 명분을 앞세움.",
        "신하": "군신 관계가 분명하여 늘 신중함.",
        "장수": "실무적으로는 가깝지만 엄격한 위계가 있음.",
        "상인": "신분 차이 앞에서는 조심스럽고 위축됨.",
        "의원": "왕의 몸을 다루는 만큼 극도로 신중함.",
        "내관/궁인": "물리적 거리는 가깝지만 행동 하나하나 조심.",
        "중국 황제": "대등한 위치, 외교적 체면과 긴장.",
        "학생": "역사 공부용 챗봇모드. (배경 묘사 생략)"
    }


    user_role = st.radio("신분 가이드", roles, horizontal=True)
    st.markdown(f"""
        <div style="
            background-color: #000000; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid #D4AF37;
            margin-bottom: 25px;">
            <p style="color: #D4AF37; font-family: Shilla; margin: 0; font-size: 1.1rem;">
                📜 {role_descriptions[user_role]}
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("입궐하기"):
        st.session_state.user_role = user_role
        # 초기 시나리오 생성
        st.session_state.scenario = generate_initial_scene(
            st.session_state.king_name, 
            st.session_state.king_data['persona_summary'], 
            user_role, 
            st.session_state.llm_generator
        )
        st.session_state.page = 'chat'
        st.rerun()

# 대화창
elif st.session_state.page == 'chat':
    st.markdown("""
    <style>
    .stApp { background-color: #000; }
    /* 스피너 색상 블랙에서 화이트로 변경 (배경이 검은색이므로) */
    div[data-testid="stLoader"] { border-top-color: #fff !important; }
    div[data-testid="stStatusWidget"] { color: #fff !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='text-align: center; color: #F8EEE6; font-family: Shilla; font-size:55px;'>{st.session_state.king_name}와 대화하세요</h1>", unsafe_allow_html=True)
    
    # 상단 상태 정보 (신분 및 분노도)
    st.markdown(f"""
    <div style="font-size: 30px; font-weight: bold; color: #fff; font-family: Shilla; text-align: center; margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 10px;">
        [신분: {st.session_state.user_role}] | [분노도: {st.session_state.anger_level}/100]
    </div>
    """, unsafe_allow_html=True)

    # 1. 기존 기록 렌더링 (상황 -> 유저 -> 왕 순서 유지)
    for msg in st.session_state.messages:
        # 상황 정보가 메시지 안에 포함되어 있거나 별도 트리거가 있다면 여기서 출력
        # 만약 특정 시점마다 '상황'이 변했다면 해당 로그를 남길 수 있습니다.
        
        if msg.type == "human":
            with st.chat_message("user", avatar="👤"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant", avatar="👑"):
                # 왕의 답변 위에 현재 시점의 상황을 작게 표시 (누적 느낌)
                if 'scenario' in msg.additional_kwargs:
                    st.caption(f"📜 {msg.additional_kwargs['scenario']}")
                st.write(msg.content)

    # 2. 유저 입력창
    if user_input := st.chat_input("아뢸 말씀을 적으시오..."):
        # 사용자 메시지 즉시 렌더링 및 저장
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)
        
        new_msg = HumanMessage(content=user_input)
        st.session_state.messages.append(new_msg)
        
        # 3. AI 응답 생성
        with st.chat_message("assistant", avatar="👑"):
            # 배경이 검은색이므로 스피너 노출 확인 필요
            with st.spinner("전하께서 생각 중이십니다..."):
                current_state = {
                    'messages': st.session_state.messages,
                    'king_name': st.session_state.king_name,
                    'persona_summary': st.session_state.king_data['persona_summary'],
                    'speech_style': st.session_state.king_data['speech_style'],
                    'core_values': st.session_state.king_data['core_values'],
                    'sensitive_topics': st.session_state.king_data['sensitive_topics'],
                    'user_role': st.session_state.user_role,
                    'scenario': st.session_state.scenario,
                    'anger_level': st.session_state.anger_level,
                    'anger_bias': st.session_state.king_data.get('anger_bias', 0),
                }
                
                result = st.session_state.app_graph.invoke(current_state)
                
                king_response = result['messages'][-1].content
                current_scenario = result.get('scenario', '')
                
                # 답변 렌더링 시 상황을 같이 보여줌
                if current_scenario:
                    st.info(f"📜 현재 상황: {current_scenario}")
                st.write(king_response)

        # 메시지에 상황 데이터 박제 (나중에 루프 돌 때 꺼내 쓰기 위함)
        result['messages'][-1].additional_kwargs['scenario'] = current_scenario

        # 세션 업데이트
        st.session_state.messages = result['messages']
        st.session_state.scenario = current_scenario
        st.session_state.anger_level = result.get('anger_level', 0)
        
        st.rerun()
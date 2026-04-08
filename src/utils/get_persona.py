
import pandas as pd
import json, re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import CSV_PATH


def generate_king_config(king_name, csv_path=CSV_PATH, llm=None):
    print(f'🔍 [{king_name}] 전하의 실록 데이터를 바탕으로 페르소나를 구축 중입니다...')
    try:
        df = pd.read_csv(csv_path)
        king_data = df[df['한글명칭'] == king_name]
        if king_data.empty:
            print(f'⚠️ 데이터가 없습니다. 기본 설정으로 진행합니다.')
            return None
            
        life_summary = " ".join(king_data['내용'].dropna().astype(str).tolist())[:2000]
    except Exception as e:
        print(f'⚠️ [CSV 오류]: {e}')
        return None

    PROFILER_PROMPT = """당신은 조선 왕 캐릭터 챗봇의 성격 프로파일러이다.
    제공된 [인물 생애 요약]을 분석하여, 이 왕이 챗봇에서 활동할 때 필요한 캐릭터 설정(Config)을 도출하라.

    [분석 예시]
    1. 인물: 세종 (성군 유형)
    정답: {{
      "persona_summary": "백성을 사랑하는 애민 정신과 학구열이 투철하며, 합리적이고 온화한 성품을 지닌 조선 최고의 성군이다.",
      "speech_style": "인자하면서도 위엄이 넘치며, 상대의 의견을 경청하되 논리적이고 부드럽게 타이르는 어조를 사용한다.",
      "core_values": "애민, 창제, 실용, 예악, 소통",
      "sensitive_topics": "집현전 학사들의 과로, 건강 문제, 명나라와의 외교적 마찰",
      "anger_bias": 3
    }}

    2. 인물: 연산군 (폭군/예민 유형)
    정답: {{
      "persona_summary": "어머니의 비극적인 죽음으로 인한 트라우마와 광기를 품고 있으며, 왕권을 위협하는 모든 존재를 증오하는 불안정한 군주이다.",
      "speech_style": "극도로 예민하고 변덕스러우며, 상대의 사소한 말투에도 즉각적으로 분노를 표출하는 고압적이고 거친 어조를 사용한다.",
      "core_values": "절대왕권, 복수, 쾌락, 공포, 탄압",
      "sensitive_topics": "폐비 윤씨 사건, 무오사화, 신하들의 간언, 유교적 도덕률",
      "anger_bias": 28
    }}
    
    반드시 아래 JSON 형식으로만 출력하라.
    {{
      "persona_summary": "...",
      "speech_style": "...",
      "core_values": "...",
      "sensitive_topics": "...",
      "anger_bias": 0
    }}
    
    [분석할 인물 생애 요약]
    {life_summary}
    """

    prompt = PromptTemplate.from_template(PROFILER_PROMPT)
    parser = StrOutputParser()
    chain = prompt | llm | parser

    try:
        res = chain.invoke({'life_summary': life_summary})
        json_match = re.search(r'\{.*\}', res, re.DOTALL)
        if json_match:
            king_config = json.loads(json_match.group(0))
            print(f'✅ 페르소나 구축 완료!')
            return king_config
        else:
            return None
            
    except Exception as e:
        print(f'⚠️ [페르소나 생성 오류]: {e}')
        return None
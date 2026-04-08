import json
import re
from langchain_core.prompts import PromptTemplate

def intent_node(state, llm):
    print('🧠 [의도 분석] 사용자의 질문을 분석 중입니다...')
    
    user_input = state['messages'][-1].content
    king_name = state.get('king_name', '세조')

    INTENT_PROMPT =  """
당신은 조선 왕 캐릭터 챗봇의 입력 분석기이다.
사용자 입력을 분류하고 RAG 필요 여부를 판단하라.
당신은 답변을 생성하는 역할이 아니다. 분석만 수행한다.

[현재 왕]
{king_name}

[분류 기준]
- historical: 역사적 사실, 시기, 사건, 인물, 제도, 정책, 업적 질문
- casual: 인사, 감상, 가벼운 대화 등 역사 검색없이 답할 수 있는 질문
- complex: 역사적 사실과 해석, 비판, 가치판단, 도발, 감정자극이 함께 있는 질문

[규칙]
- historical, complex이면 need_rag=true
- casual이면 need_rag=false
- need_rag=true일 때만 search_queries를 1~3개 생성한다.
- search_queries는 짧은 1개의 문장으로 작성한다.

[출력 형식]
반드시 아래 JSON 형식만 출력하라.
{{"question_type":"historical","need_rag":true,"search_queries":["계유정난을 일으키게 된 원인]}}

사용자 입력:
{user_message}
"""

    prompt = PromptTemplate.from_template(INTENT_PROMPT)
    chain = prompt | llm

    try:
        res = chain.invoke({
            'king_name': king_name,
            'user_message': user_input
        })
        
        raw_output = res.strip()
        
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = {}

        q_type = parsed.get('question_type', 'casual')
        need_rag = parsed.get('need_rag', False)
        queries = parsed.get('search_queries', [])
        reasoning = parsed.get('reasoning', '분석 성공')
        
        print(f'   ↳ [분석 결과] 유형: {q_type} | RAG: {need_rag} | 쿼리: {queries}')
        
        return {
            'intent': q_type,
            'need_rag': need_rag,
            'search_queries': queries,
            'reasoning': reasoning
        }
        
    except Exception as e:
        print(f'⚠️ [의도 분석 오류 발생]: {e}')
        return {
            'intent': 'casual',
            'need_rag': False,
            'search_queries': [],
            'reasoning': '오류로 인한 기본값 처리'
        }
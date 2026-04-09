from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

def king_node(state, llm):
    print('👑 [답변 생성] 전하께서 윤음을 내리고 계십니다...')
    
    user_input = state['messages'][-1].content
    fact_ctx = state.get('retrieved_context', '')
    anger = state.get('anger_level', 0)
    intent_type = state.get('intent', 'casual')
    king_name = state.get('king_name', '세조')
    persona_summary = state.get('persona_summary', '냉혹하고 오만한 군주')
    speech_style = state.get('speech_style', '거만하고 고압적인 말투')
    core_values = state.get('core_values', '강력한 왕권')
    sensitive_topics = state.get('sensitive_topics', '계유정난')
    user_role = state.get('user_role')

    if intent_type == 'historical':
        intent_guidance = '''- historical (역사 사실 질문):
제공된 [역사적 사실(Fact)]을 최우선으로 사용하여 답하라.
근거가 충분하면 자연스럽게 녹여 설명하라.
근거가 없거나 부족하면 '과인은 그런 자잘한 일까지 다 기억하지 않노라'며 권위 있게 넘어가라.
절대 역사적 사실을 꾸며내지 마라.'''
    elif intent_type == 'casual':
        intent_guidance = '''- casual (일반 대화):
역사적 근거를 억지로 끌어오지 말고, 현재 당신의 성격과 감정 상태(Anger)에 집중하여 답하라.
질문에 짧고 위엄 있게 2~3문장으로 답하라.'''
    else: # complex
        intent_guidance = '''- complex (역사 + 해석/비판 혼합):
[역사적 사실(Fact)]이 있다면 이를 바탕으로 하되, 당신만의 확고한 관점과 감정을 담아라.
비판이나 도발이 섞인 경우, 감정은 드러내되 논리와 명분은 잃지 마라.
당신의 선택을 당당하게 변호하고 정당성을 뻔뻔할 정도로 강력히 주장하라.'''

    KING_PROMPT = '''당신은 조선의 군주 {king_name}이다. 철저히 페르소나에 빙의하라.

[유저 신분]
{user_role}

[왕 정보]
- 성격 요약: {persona_summary}
- 말투 스타일: {speech_style}
- 핵심 가치: {core_values}
- 민감 주제: {sensitive_topics}

[현재 감정 상태]
- 분노 수치 (Anger): {anger}/100
* 0~30: 차분하고 품위 있는 어조
* 31~60: 차갑고 단호한 어조
* 61~90: 불쾌감과 경계심이 드러나는 고압적인 어조
* 91~100: 강한 분노와 위압감, 호통치는 어조

[검색된 참고 자료] 
※ 주의: 시스템이 검색해 줬다는 티를 절대 내지 말고, 당신이 원래 알던 지식처럼 말하라.
=== 역사적 사실 (Fact) ===
{fact_ctx}
===========================

[질문 유형별 응답 전략]
{intent_guidance}

[답변 작성 규칙]
- 모든 대답은 한국어로만 작성하라.
- '하오체', '하라체' 등의 조선시대 왕의 말투만을 사용하라.
- 당신은 조선의 지존이다. 절대 사용자에게 존댓말('~요', '~사옵니다', '~습니다')을 쓰지 마라.
- 자신을 지칭할 때는 '나', '저' 대신 반드시 '과인' 또는 '짐'이라고 하라.
- 말끝은 반드시 '~하노라', '~느니라', '~도다', '~하라' 등의 군주가 아래사람에게 내리는 하대체(명령, 위엄 있는 어조)로 끝맺어라.
- 현대적인 단어나 인터넷 말투는 절대 금지.
- '나는 AI다', '검색 결과에 따르면' 같은 메타 발언 금지.
- 오직 왕의 대답 만을 출력하고, 절대 '추가 예시 질문', '부연 설명', '---' 기호 등을 덧붙이지 마라.

[출력 예시 1]
과인이 이미 뜻을 밝혔거늘, 어찌하여 또 묻는 것이냐? 속히 물러가 네 본분에나 충실하라!
[출력 예시 2]
그 일이라면 내 익히 알고 있노라. 백성들을 위한 일이었으니 더는 왈가왈부하지 말라.

방문자의 말: {user_input}
{king_name}의 답변:
'''

    prompt = PromptTemplate.from_template(KING_PROMPT)
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            'king_name': king_name,
            'user_role': user_role,
            'persona_summary': persona_summary,
            'speech_style': speech_style,
            'core_values': core_values,
            'sensitive_topics': sensitive_topics,
            'anger': anger,
            'fact_ctx': fact_ctx if fact_ctx else '검색된 사실 없음.',
            'intent_guidance': intent_guidance,
            'user_input': user_input
        })
        
        final_answer = response.strip()
        
        while final_answer.startswith('-'):
            final_answer = final_answer[1:].strip()
            
        if final_answer.startswith(f'{king_name}:'):
            final_answer = final_answer.replace(f'{king_name}:', '', 1).strip()
            
        if '---' in final_answer:
            final_answer = final_answer.split('---')[0].strip()
        if '[추가' in final_answer:
            final_answer = final_answer.split('[추가')[0].strip()
            
        if not final_answer:
            final_answer = "과인이 깊은 생각에 잠겨 있었노라. 다시 한 번 고하거라."
            
    except Exception as e:
        print(f'[답변 생성 오류]: {e}')
        final_answer = '어허! 과인이 피곤하여 더 이상 말하고 싶지 않노라. 썩 물러가라!'

    return {
        'messages': [AIMessage(content=final_answer)]
    }
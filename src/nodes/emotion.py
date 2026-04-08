import re
from langchain_core.prompts import PromptTemplate

def emotion_node(state, llm):
    print('🎭 [감정 판정] 왕의 기분 변화를 계산 중입니다...')
    user_input = state['messages'][-1].content
    current_anger = state.get('anger_level', 0)
    king_name = state.get('king_name', '세조')
    core_values = state.get('core_values', '강력한 왕권과 법치')
    sensitive_topics = state.get('sensitive_topics', '단종 폐위, 계유정난')
    anger_bias = state.get('anger_bias', 5) 

    EMOTION_JUDGE_PROMPT = '''당신은 조선 왕 캐릭터 챗봇의 감정 판정기이다.
사용자의 발화를 보고, 현재 왕이 느낄 분노 변화량을 정수 하나로 판단하라.

[현재 왕 정보]
- 왕 이름: {king_name}
- 핵심 가치: {core_values}
- 민감 주제: {sensitive_topics}

[현재 분노 수치]
{current_anger}
- anger 수치는 0~100 범위이며, 최대 분노 수치는 100이다.

[기본 판정 기준]
- 매우 공손하고 존중하는 태도: -10 ~ -5
- 중립적 / 정보 요청성 질문: 0
- 약간 무례함, 빈정거림: +5 ~ +10
- 노골적 비난, 조롱, 정통성 공격: +25 ~ +30

[왕별 민감도 보정]
위 기준에서 아래 보정치를 더한 값을 최종 출력으로 삼아라.
- anger_bias: {anger_bias} (양수면 이 왕은 평균보다 더 예민, 음수면 더 관대)

[규칙]
- 민감 주제에 대한 비난, 조롱, 정통성 공격이 포함되면 높은 분노 상승치를 준다.
- 반드시 정수 하나만 출력한다.
- 설명, 이유, 다른 문장 일절 쓰지 않는다.
- 출력 예: -5 / 0 / 10 / 20

사용자 발화:
{user_input}'''

    prompt = PromptTemplate.from_template(EMOTION_JUDGE_PROMPT)
    chain = prompt | llm  
    
    try:
        res = chain.invoke({
            'king_name': king_name,
            'core_values': core_values,
            'sensitive_topics': sensitive_topics,
            'current_anger': current_anger,
            'anger_bias': anger_bias,
            'user_input': user_input
        })
        
        raw_output = res.strip()
        numbers = re.findall(r'-?\d+', raw_output)
        
        if numbers:
            delta = int(numbers[0])
        else:
            delta = 0
            
    except Exception as e:
        print(f'⚠️ [감정 판정 오류]: {e}')
        delta = 0

    new_anger = current_anger + delta
    new_anger = max(0, min(100, new_anger))
    
    sign = '+' if delta > 0 else ''
    print(f'   ↳ [감정 변화] {sign}{delta} ➡️ 현재 분노 수치: {new_anger}/100')

    is_max_anger = (new_anger >= 100)
    
    return {
        'anger_level': new_anger,
        'is_max_anger': is_max_anger
    }
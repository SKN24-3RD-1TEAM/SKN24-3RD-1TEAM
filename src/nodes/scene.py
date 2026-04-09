from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def scene_node(state, llm):
    king_response = state['messages'][-1].content
    user_input = state['messages'][-2].content
    persona_summary = state['persona_summary']
    king_name = state['king_name']
    user_role = state['user_role']
    anger_level = state['anger_level']
    if user_role != '학생':
        print(f'🎬 [배경 묘사] 직전 대화에 맞게 배경을 변화하는 중입니다...')

    if user_role == '학생' or not king_response:
        return {'scenario': ''}
    
    SCENE_PROMPT = '''
당신은 조선 시대 역사 인물 몰입형 챗봇의 장면 묘사기이다.

[지시사항]
- 모든 내용은 한국어로만 작성하라.
- 조선 시대 분위기가 느껴지는 문체로 쓴다.
- 전하께서 대답을 마친 직후의 분위기, 표정, 행동, 편전의 정적 등을 1~2문장으로 묘사하라.
- 방금 오간 대사의 내용을 그대로 반복하지 말고, 그 말이 남긴 여파와 공기의 변화를 보여준다.
- 왕의 성격, 권위, 감정 변화, 사용자와의 거리감이 드러나야 한다.
- 신분 차이 원칙을 고려하여 장면을 생성한다.
- 절대 대사, 설명, 제목, 따옴표를 출력하지 않는다.

[상황 정보]
- 왕 이름: {king_name}
- 왕의 성격: {persona_summary}
- 사용자 신분: {user_role}

[직전 대화]
- 방문자의 말: {user_input}
- 전하의 답변: {king_response}

[신분 차이 원칙]
- 백성: 극도의 긴장과 두려움, 위축이 강하게 드러나야 한다.
- 유생: 예를 지키면서도 명분과 소신의 기류가 느껴질 수 있다.
- 신하: 군신 관계 속 신중함과 압박감이 드러나야 한다.
- 장수: 실무적 거리감 속에서도 엄한 위계와 군무의 긴장감이 드러나야 한다.
- 상인: 왕 앞에서 위축되고 조심스러운 분위기가 드러나야 한다.
- 의원: 왕 가까이에 있으나 극도로 신중한 반응이 드러나야 한다.
- 내관/궁인: 물리적으로 가깝지만 작은 반응 하나도 조심스러운 분위기가 드러나야 한다.
- 중국 황제: 외교적 체면과 긴장, 상호 견제가 드러나야 하며 일방적 복종처럼 쓰지 않는다.
- 학생: 조선 시대 실제 신분이 아니므로 장면 묘사를 생성하지 않는다. 반드시 빈 문자열만 출력한다. 

- [절대 금지 사항]
  1. 어떠한 경우에도 대사나 인용문을 만들지 마라. 
  2. 따옴표(" " 또는 ' ')를 절대 사용하지 마라.
  3. 오직 시각적 묘사(표정, 몸짓)와 공간의 분위기만 서술하라.

장면 묘사:
'''

    prompt = PromptTemplate.from_template(SCENE_PROMPT)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        scenario_text = chain.invoke({
            'king_name': king_name,
            'persona_summary': persona_summary,
            'user_role': user_role,
            'king_response': king_response,
            'user_input': user_input
        }).strip()
        
        if '장면 묘사:' in scenario_text:
            scenario_text = scenario_text.split('장면 묘사:')[-1].strip()
            
        lines = [line.strip() for line in scenario_text.split('\n') if line.strip()]
        
        if lines:
            scenario_text = lines[0]
        else:
            scenario_text = ''
            
        return {'scenario': scenario_text}
        
    except Exception as e:
        print(f'⚠️ [초기 배경 생성 오류]: {e}')
        return {'scenario': f'무거운 침묵 속, {king_name} 전하께서 옥좌에서 당신을 내려다보고 계신다.'}
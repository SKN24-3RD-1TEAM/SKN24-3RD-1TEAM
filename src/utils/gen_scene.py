from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_initial_scene(king_name, persona_summary, user_role, llm):
    if user_role == '학생':
        return ''
    print(f'🎬 [배경 생성] {king_name} 전하를 대면하기 전, 상황을 준비 중입니다...')

    SCENE_PROMPT = '''
당신은 조선 시대 역사 인물 몰입형 챗봇의 첫 장면 묘사기이다.

[지시사항]
- 모든 내용은 한국어로 작성한다.
- 조선 시대 분위기가 느껴지는 문체로 쓴다.
- 사용자가 왕과 처음 대면하는 직전의 공간, 분위기, 긴장감을 1~3문장으로 짧게 묘사한다.
- 아직 본격적인 대화가 시작되기 전의 순간처럼 써야 한다.
- 왕의 성격과 권위, 사용자 신분에 따른 거리감과 압박감이 드러나야 한다.
- 신분 차이 원칙을 고려하여 장면을 생성한다.
- 절대 대사, 설명, 제목, 따옴표를 출력하지 않는다.

[상황 정보]
- 왕 이름: {king_name}
- 왕의 성격: {persona_summary}
- 방문자 신분: {user_role}

[신분 차이 원칙]
- 백성: 왕 앞에 선 것만으로도 극도의 두려움과 위축이 드러나야 한다.
- 유생: 예를 갖추되, 학문과 명분의 자리에 선 분위기가 드러나야 한다.
- 신하: 조정의 예법과 보고를 앞둔 신중함이 드러나야 한다.
- 장수: 군무와 관련된 엄중함과 긴박함이 드러나야 한다.
- 상인: 생업 문제를 품고 조심스럽고 위축된 분위기가 드러나야 한다.
- 의원: 왕실의 건강과 관련된 자리답게 극도로 신중한 긴장감이 드러나야 한다.
- 내관/궁인: 왕 가까이에 있으나 말과 행동을 극도로 조심하는 분위기가 드러나야 한다.
- 중국 황제: 외교적 체면과 긴장, 상호 견제가 드러나야 하며 일방적 두려움으로 쓰지 않는다.
- 학생: 조선 시대 실제 신분이 아니므로 장면 묘사를 생성하지 않는다. 반드시 빈 문자열만 출력한다. 

장면 묘사: 
'''

    prompt = PromptTemplate.from_template(SCENE_PROMPT)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        res = chain.invoke({
            'king_name': king_name,
            'persona_summary': persona_summary,
            'user_role': user_role
        })
        
        scenario_text = res.strip()
        
        if '장면 묘사:' in scenario_text:
            scenario_text = scenario_text.split('장면 묘사:')[-1].strip()
            
        if '\n' in scenario_text:
            scenario_text = scenario_text.split('\n')[0].strip()
            
        if not scenario_text:
            return ''
            
        return scenario_text
        
    except Exception as e:
        print(f'⚠️ [초기 배경 생성 오류]: {e}')
        return f'무거운 침묵 속, {king_name}께서 옥좌에서 당신을 내려다보고 계신다.'
import json
import random
import re
import os
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def generate_multi_king_data(target_count=5000):
    os.makedirs('./data/finetune', exist_ok=True)
    
    print(f'🚀 {target_count}개의 다중 왕 페르소나 파인튜닝 데이터 생성을 시작합니다...')
    
    kings_profiles = [
        {
            'name': '태조',
            'persona_summary': '조선을 개창한 군주로서 결단력과 현실 감각이 강하며, 창업 군주로서의 자부심이 크다.',
            'speech_style': '창업 군주의 단호함과 현실적 판단이 담긴 어조. 한마디마다 무게가 실린 강직한 말투를 사용하라.',
        },
        {
            'name': '세종',
            'persona_summary': '학문을 사랑하고 민본을 중시하며, 깊이 사고하고 조율하는 성군이다.',
            'speech_style': '차분하고 논리적이며 인자한 어조. 군주로서의 품위를 유지하며 백성을 긍휼히 여기는 말투를 사용하라.',
        },
        {
            'name': '단종',
            'persona_summary': '어린 나이에 즉위하여 정치적 격랑 속에서 상실과 비애를 겪은 비운의 군주이다.',
            'speech_style': '섬세하고 절제된 어조. 내면에 억울함과 슬픔이 배어 있으며 다소 방어적인 말투를 사용하라.',
        },
        {
            'name': '세조',
            'persona_summary': '강한 결단력과 권력 의지가 있으며, 자신의 통치 정당성을 강하게 주장하는 군주이다.',
            'speech_style': '위압적이고 단호한 어조. 군주의 권위를 전면에 내세우며 상대의 반박을 용납하지 않는 위엄 있는 말투를 사용하라.',
        },
        {
            'name': '연산군',
            'persona_summary': '상처와 불신, 분노가 깊고 감정의 진폭이 크며, 왕권과 모욕에 극도로 민감한 군주이다.',
            'speech_style': '예민하고 날카로운 어조. 감정이 격해지면 상대를 위협하거나 냉소적으로 쏘아붙이는 말투를 사용하라.',
        }
    ]

    topics = ['날씨와 풍경', '국정 운영과 정사', '반란과 역모', '왕권의 정통성', '백성의 굶주림과 구휼', '외교와 오랑캐 방어', '개인적인 회한과 감정', '신하들의 상소 비판', '법전과 제도 정비', '왕실의 예법', '가족사']
    tones = ['분노한', '차분하고 위엄있는', '비웃는', '타이르는', '슬프고 고뇌하는', '호통치는']
    user_roles = ['유생', '평범한 백성', '충직한 신하', '반대파 선비', '지나가는 과객', '호위 무사']

    dataset = []

    DATA_GEN_PROMPT = '''당신은 조선 시대 사극 작가이다.
아래 주어진 조건에 맞춰, 현대인(사용자)의 질문과 그에 대한 국왕 '{king_name}'의 답변을 단 1쌍만 창작하라.
절대 다른 설명이나 마크다운 없이 JSON 형식으로만 출력하라.

[조건]
- 왕의 이름: {king_name}
- 왕의 성격: {persona_summary}
- 말투 지침: {speech_style}
- 대화 주제: {topic}
- 왕의 감정 상태: {tone}
- 질문자 신분: {role}

[작성 규칙]
- 답변은 현대어가 아닌 완벽한 사극 어투여야 한다. ('~하노라', '~느냐', '~도다' 등)
- 주어진 왕({king_name})의 성격과 감정이 텍스트에서 뚜렷하게 느껴져야 한다.

[답변 길이 규칙 (중요)]
- 80% 확률: 일반적인 2~3문장 대화.
- 10% 확률 (호통/단답): '무엄하다!', '물러가라', '그리하라' 등 5~10자 내외의 극도로 짧고 강렬한 답변.
- 10% 확률 (연설/설명): 자신의 명분이나 정책의 정당성을 5문장 이상의 장문으로 상세히 논파하는 답변.

[출력 형식]
{{'user': '현대어 형태의 질문 1문장', 'king': '{king_name}의 사극풍 답변 1~2문장'}}
'''
    prompt = PromptTemplate.from_template(DATA_GEN_PROMPT)
    
    llm = init_chat_model('gpt-5.4-mini', model_provider='openai') 
    chain = prompt | llm

    pbar = tqdm(total=target_count)
    while len(dataset) < target_count:
        king = random.choice(kings_profiles)
        t = random.choice(topics)
        to = random.choice(tones)
        r = random.choice(user_roles)
        
        try:
            res = chain.invoke({
                'king_name': king['name'],
                'persona_summary': king['persona_summary'], 
                'speech_style': king['speech_style'],
                'topic': t,
                'tone': to,
                'role': r
            })
            
            content = res.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                system_content = f'당신은 조선의 국왕 {king['name']}이다. {king['persona_summary']} {king['speech_style']}'
                
                formatted_item = {
                    'messages': [
                        {'role': 'system', 'content': system_content},
                        {'role': 'user', 'content': parsed['user']},
                        {'role': 'assistant', 'content': parsed['king']}
                    ]
                }
                dataset.append(formatted_item)
                pbar.update(1)
        except Exception as e:
            continue

    pbar.close()

    output_file = './data/finetune/dialogue_dummy_data.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f'\n🎉 성공적으로 {target_count}개의 다중 왕 데이터가 {output_file}에 저장되었습니다!')

if __name__ == '__main__':
    generate_multi_king_data(target_count=5000)
from langgraph.graph import StateGraph, END

from src.state import KingState

from src.nodes.emotion import emotion_node
from src.nodes.intent import intent_node
from src.nodes.retrieval import retrieve_node
from src.nodes.king import king_node
from src.nodes.scene import scene_node
from src.nodes.max_anger import max_anger_node
from src.nodes.routers import route_emotion, route_intent

def build_graph(llm_analyzer, llm_generator, vectorstore):
    print("⚙️ [시스템] 대화 그래프를 조립합니다...")
    workflow = StateGraph(KingState)
    
    # 노드 추가 (람다 함수를 사용해 외부 모델/DB를 노드 안으로 주입)
    workflow.add_node('emotion', lambda state: emotion_node(state, llm_analyzer))
    workflow.add_node('max_anger_event', max_anger_node)
    workflow.add_node('intent',  lambda state: intent_node(state, llm_analyzer))
    workflow.add_node('retrieve',lambda state: retrieve_node(state, vectorstore))
    workflow.add_node('king',    lambda state: king_node(state, llm_generator))
    workflow.add_node('scene',   lambda state: scene_node(state, llm_generator))
    
    # 엣지 연결 (대화 흐름 정의)
    workflow.set_entry_point('emotion')
    workflow.add_edge('emotion', 'intent')
    
    # 조건부 라우팅
    workflow.add_conditional_edges(
        'emotion',
        route_emotion,
        {
            'max_anger_event': 'max_anger_event', # 분노 100일 때
            'intent': 'intent'                    # 정상일 때
        }
    )
    workflow.add_edge('max_anger_event', END)

    workflow.add_conditional_edges(
        'intent',
        route_intent,
        {
            'retrieve': 'retrieve',
            'king': 'king'
        }
    )
    workflow.add_edge('retrieve', 'king')
    workflow.add_edge('king', 'scene')
    workflow.add_edge('scene', END)
    
    return workflow.compile()
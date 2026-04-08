from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class KingState(TypedDict):
    '''LangGraph의 모든 노드가 공유할 '장부(메모리)'입니다.'''
    messages: Annotated[list[BaseMessage], add_messages]
    king_name: str           # 예: '세조'
    scenario: str            # 예: '경복궁 근정전에서 정무를 보던 중'
    persona_summary: str     # 예: '피바람을 일으키고 왕위를 쟁취한 절대 군주'
    speech_style: str        # 예: '매우 거만하고 폭력적이며 억압적'
    core_values: str         # 예: '강력한 왕권, 법치'
    sensitive_topics: str    # 예: '계유정난, 단종 사사'
    user_role: str           # 예: '백성'
    anger_level: int         # 0 ~ 100
    intent: str              # 'historical', 'casual', 'complex'
    need_rag: bool           # True / False
    search_queries: List[str]# ['경국대전', '세조']
    reasoning: str           # 분석 이유
    retrieved_context: str   # 역사적 사실(Fact) 텍스트
    is_max_anger: bool       # True / False
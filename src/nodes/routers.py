def route_intent(state):
    """의도 분석 결과에 따라 RAG 검색 노드로 갈지, 답변 노드로 갈지 결정"""
    if state.get('need_rag', False):
        return 'retrieve'
    return 'king'

def route_emotion(state):
    """분노가 100이면 특별 이벤트로, 아니면 정상적인 의도 분석으로 이동"""
    if state.get('is_max_anger', False):
        return 'max_anger_event'
    return 'intent'
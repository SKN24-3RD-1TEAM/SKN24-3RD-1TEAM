def retrieve_fact(queries: list, need_rag: bool, vectorstore):
    '''
    RAG 필요 여부에 따라 실록 및 역사 사실 데이터만 검색합니다.
    vectorstore를 인자로 받아 실행됩니다.
    '''
    fact_text = ''

    if need_rag and queries:
        fact_docs = []
        for q in queries:
            try:
                docs = vectorstore.similarity_search(
                    q, 
                    filter={'document_type': {'$in': ['sillok', 'history_non', 'history_person']}}, 
                    k=2
                )
                fact_docs.extend(docs)
            except Exception as e:
                print(f'⚠️ [DB 검색 오류]: {e}')
        
        if fact_docs:
            unique_facts = list(dict.fromkeys([doc.page_content for doc in fact_docs]))
            fact_text = '\n\n'.join(unique_facts[:3])

    return fact_text


def retrieve_node(state, vectorstore):
    """
    LangGraph의 검색 노드 함수입니다.
    """
    print('📚 [정보 검색] 실록과 역사 데이터를 찾고 있습니다...')
    
    need_rag = state.get('need_rag', False)
    search_queries = state.get('search_queries', [])
    
    fact_ctx = retrieve_fact(search_queries, need_rag, vectorstore)
    
    if fact_ctx:
        print('   ↳ 📜 팩트 검색 완료 (RAG 적용됨)')
        print('   ↳ 🔍 [검색된 원문 미리보기]:')
        preview_text = fact_ctx.replace('\n', ' ')
        print(f"      {preview_text[:150]}{'...' if len(fact_ctx) > 150 else ''}")
    elif need_rag:
        print('   ↳ ⚠️ 검색 조건에 맞는 역사적 기록을 찾지 못했습니다.')
    
    return {'retrieved_context': fact_ctx}
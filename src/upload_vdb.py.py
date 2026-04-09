import pandas as pd
import chromadb
import json
from tqdm import tqdm
from chromadb.utils import embedding_functions
import re
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

class MyEmbeddings(HuggingFaceEmbeddings):
    name: str = "kure_v1"

embedding_function = MyEmbeddings(
    model_name="nlpai-lab/KURE-v1",
    model_kwargs={'device': 'cuda'}
)

text_splitter = SemanticChunker(
    embedding_function, 
    breakpoint_threshold_type="percentile", 
    breakpoint_threshold_amount=95 
)

chroma_client = chromadb.PersistentClient(path="../../database")

collection = chroma_client.get_or_create_collection(
    name="reference", 
    embedding_function=embedding_function
)

regex = r'\([^)]*[\u4e00-\u9fff][^)]*\)'
batch_size = 322

kings = ['태조', '세종', '단종', '세조', '연산군']

for king in kings:
    print(f"🚀 {king}실록 데이터 처리 및 업로드 중...")
    with open(f'{king}실록.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current_king_data = []
    # 데이터 파싱
    for k_name, books in data.items():
        for book, days in books.items():
            for day, articles in days.items():
                for contents in articles:
                    article_num = contents['article_num']
                    part = article_num.split('-')
                    
                    title = contents['title']
                    clean_content = contents['content'].split('【')[0].strip()
                    category = contents['content'].split('【분류】')[1].strip() if '【분류】' in contents['content'] else ""

                    text = re.sub(regex, '', f"{title}\n{clean_content}")
                    
                    current_king_data.append({
                        "text": text,
                        "metadata": {
                            "regnal_year": part[0],
                            "month": part[1],
                            "day": part[2].split('[')[0],
                            "article_num": article_num,
                            "category": category,
                            "king_name": king,
                            "document_type": 'sillok'
                        }
                    })
    
    # 해당 왕의 데이터가 모이면 즉시 배치 업로드
    for i in tqdm(range(0, len(current_king_data), batch_size)):
        batch = current_king_data[i : i + batch_size]
        collection.upsert(
            ids=[f"{doc['metadata']['article_num']}_{idx}" for idx, doc in enumerate(batch, start=i)],
            documents=[doc['text'] for doc in batch],
            metadatas=[doc['metadata'] for doc in batch]
        )

print(f"🚀 역사 데이터(History) 업로드 중...")
history_files = [
    ('history_non_person_data.csv', 'history_non'),
    ('history_person_data.csv', 'history_person')
]

history_data_chunks_list = []

for file_path, doc_type in history_files:
    df = pd.read_csv(file_path)
    history_data = [] 

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {doc_type}"):
        combined_text = f"{row['한글명칭']}\n{row['내용']}"
        combined_text = re.sub(regex, '', combined_text).strip()

        chunks = text_splitter.split_text(combined_text)
        
        for i, chunk in enumerate(chunks):

             history_data.append({
                "text": f"{row['한글명칭']}\n{chunk}", 
                "metadata": {
                    "document_type": doc_type,
                    "article_num": row['한글명칭']
                }
            })


    for i in tqdm(range(0, len(history_data), batch_size)):
        batch = history_data[i : i + batch_size]
        collection.upsert(

            ids=[f"{doc['metadata']['article_num']}_{idx}" for idx, doc in enumerate(batch, start=i)],
            documents=[doc['text'] for doc in batch],
            metadatas=[doc['metadata'] for doc in batch]
        )

print("✅ 역사 데이터 작업 끝!")
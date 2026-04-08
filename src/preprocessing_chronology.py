import pandas as pd
import re

def clean_hanja_and_brackets(text):
    '''
    텍스트 내의 한자와 관련된 괄호를 제거하는 함수입니다.
    '''
    if pd.isna(text):
        return text
    
    text = str(text)
    text = re.sub(r'\([^)]*[\u4e00-\u9fff]+[^)]*\)', '', text)
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_history_data(file_path):
    df = pd.read_excel(file_path)
    if '내용' in df.columns:
        df['내용'] = df['내용'].apply(clean_hanja_and_brackets)
        
    if '설명' in df.columns:
        df['설명'] = df['설명'].apply(clean_hanja_and_brackets)
    
    target_columns = ['유형', '한글명칭', '설명', '내용']

    existing_columns = [col for col in target_columns if col in df.columns]
    df = df[existing_columns]

    if '유형' in df.columns:
        df_person = df[df['유형'] == '인물']
        df_non_person = df[df['유형'] != '인물']
    else:
        return
    
    person_file = './data/preprocessed/history_person_data.csv'
    non_person_file = './data/preprocessed/history_non_person_data.csv'
    
    df_person.to_csv(person_file, index=False, encoding='utf-8-sig')
    df_non_person.to_csv(non_person_file, index=False, encoding='utf-8-sig')

    print(f' - 인물 데이터: {len(df_person)}건 저장 완료 ({person_file})')
    print(f' - 비인물(사건/지리 등) 데이터: {len(df_non_person)}건 저장 완료 ({non_person_file})')

if __name__ == '__main__':
    file_name = './data/raw/교육부 국사편찬위원회_우리역사넷_한국사연대기_20251203.xlsx'
    process_history_data(file_name)
import time, json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm
from src.crawl.crawl_ref import KING_INFO, LEAP_MONTH

def generate_king_codes(king_dict:dict, leap_months_dict:dict, king_name:str) -> list:
    info = king_dict.get(king_name)
    if not info:
        return []

    start_year, start_month = info['start']
    end_year, end_month = info['end']
    code_list = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue
            code_list.append(f'ITKC_JT_{info['code']}_A{year:02d}_{month:02d}A')
            
            if king_name in leap_months_dict and year in leap_months_dict[king_name]:
                if month in leap_months_dict[king_name][year]:
                    code_list.append(f'ITKC_JT_{info['code']}_A{year:02d}_{month:02d}B')

    return code_list


def crawl_daily_sillok(driver, data_id):
    url = f'https://db.itkc.or.kr/dir/item?itemId=JT#/dir/node?dataId={data_id}'
    articles = []
    
    try:
        old_h4 = driver.find_element(By.CSS_SELECTOR, 'div.text_body_tit h4')
    except:
        old_h4 = None

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 1.5)
        
        if old_h4:
            try:
                wait.until(EC.staleness_of(old_h4))
            except TimeoutException:
                pass
                
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.text_body_tit h4')))

        container_elements = driver.find_elements(By.CSS_SELECTOR, 'div.text_body_scroll > div')
        for container in container_elements:
            try:
                h4_elem = container.find_element(By.CSS_SELECTOR, 'div.text_body_tit h4')
                article_num = h4_elem.find_element(By.CLASS_NAME, 'datenum').text.strip()
                pure_title = h4_elem.text.replace(article_num, '').strip()

                body_elements = container.find_elements(By.CSS_SELECTOR, 'div.text_body > div')
                content = "\n".join([e.text for e in body_elements if e.text.strip()])

                if content:
                    articles.append({
                        'article_num': article_num,
                        'title': pure_title,
                        'content': content
                    })
            except Exception: 
                continue
                
        return articles
        
    except TimeoutException:
        return []
    except Exception as e:
        return []
    

def start_sillok_crawling(king_name, king_dict, leap_dict):
    month_codes = generate_king_codes(king_dict, leap_dict, king_name)
    if not month_codes: 
        return

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    full_sillok_data = {}
    final_filename = f'{king_name}실록.json'

    print(f'{king_name} 실록 수집을 시작')

    pbar_month = tqdm(month_codes, desc="전체 월 진행도", unit="월")

    for m_code in pbar_month:
        parts = m_code.split('_')
        
        raw_year = parts[3] 
        year_key = f"{raw_year[1:]}년"
        
        raw_month = parts[4]
        month_num = raw_month[:-1]
        
        if raw_month.endswith('B'): # B로 끝나면 윤달
            month_key = f"윤{month_num}월"
        else:
            month_key = f"{month_num}월"

        pbar_month.set_description(f"수집 중: {year_key} {month_key}")

        if year_key not in full_sillok_data:
            full_sillok_data[year_key] = {}
        full_sillok_data[year_key][month_key] = {}

        for day in range(1, 32):
            data_id = f'{m_code}_{day:02d}A'
            
            print(f"🔍 확인 중: {data_id}...", end="\r")
            daily_articles = crawl_daily_sillok(driver, data_id)
            
            if daily_articles:
                day_key = f'{day:02d}일'
                full_sillok_data[year_key][month_key][day_key] = daily_articles
                print(f"✅ 수집 성공: {data_id}")
            time.sleep(0.1)

        with open(final_filename, 'w', encoding='utf-8') as f:
            json.dump(full_sillok_data, f, ensure_ascii=False, indent=4)

    driver.quit()
    print(f'\n{king_name} 실록 수집 완료 -> {final_filename}')

if __name__ == '__main__':
    for king in ['태조', '세종', '단종', '세조', '연산군']:
        start_sillok_crawling(king, KING_INFO, LEAP_MONTH)
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from tqdm import tqdm

import pandas as pd
import requests
import random
import time
import re


def preprocessing(headline: str, author: str, poem: str) -> str:
    """
    시의 내용을 전처리합니다.
    Args:
        headline (str): 시의 제목을 입력받습니다.
        author (str): 시를 작성한 작가를 입력받습니다.
        poem (str): 시의 내용을 입력받습니다.

    Returns:
        str: 전처리된 시의 내용을 반환합니다.
    """

    data = poem.replace("\r", "")                           # 특수문자 제거
    data = data.replace("\n \n", "\n\n")                    # 연 구분
    if data.find(author) > 0:
        data = data[data.find(author) + len(author):]       # 작가 제거
    data = data.replace(headline, "", 1).replace("/", "")   # 제목 및 / 제거
    data = re.sub(r'[^가-힣0-9\n\s]', '', data)           # 외국어 및 특수문자 제거
    data = data.strip()                                     # 양옆 \기호 제거

    data = data.replace(u"\xa0", u"")                       # "\xa0" 제거
    data = re.sub(r"( {2,})", "", data)                     # " " * m (m > 1) 제거
    data = re.sub(r"\n\n ", "", data)                       # "\n\n " 제거
    data = re.sub(r"(\n{3,})", "", data)                    # "\n" * m (m > 2) 제거
    data = re.sub(r"\n\n$", "", data)                       # 문장 끝에 위치한 "\n\n"제거

    return data


def postprocessing(df):
    """
    크롤링한 DataFrame 데이터에 대해 후처리를 수행합니다.

    Args:
        df (DataFrame): 크롤링한 DataFrame 유형의 데이터를 입력받습니다.

    Returns:
        DataFrame: 후처리된 DataFrame을 반환합니다. 
    """

    # NaN 제거
    dataset = df.dropna(subset=["poem"])
    
    # 중복 제거
    dataset = dataset.drop_duplicates(["poem"], keep="first")

    # 1000자 이상의 데이터 제거
    dataset = dataset[dataset["poem"].str.len() <= 1000]

    # index 재설정
    dataset = dataset.reset_index(drop=True) 

    # 연의 갯수를 파악하고 저장
    dataset["num_stanza"] = 0
    for i in range(len(dataset)):
        dataset.iloc[i, 3] = dataset["poem"][i].count("\n\n") + 1

    # 첫 행의 단어 길이를 파악하고 첫 행과 길이를 저장 (띄어쓰기로 구분되는 단어)
    dataset["num_line"] = 0
    dataset["first_line"] = ""
    for i in range(len(dataset)):
        text_until_newline = re.split('\n', dataset["poem"][i], 1)[0]
        word_count = len(re.findall('\s', text_until_newline)) + 1
        dataset.iloc[i, 4] = word_count
        dataset.iloc[i, 5] = text_until_newline

    return dataset


def crawling(lastest_id: int, num: int):
    """
    한국 현대 시 크롤링을 수행합니다.

    Args:
        lastest_id (int): 웹페이지에 올라온 가장 최근의 시 id를 입력받습니다.
        num (int): 수집할 데이터 수를 입력받습니다.
    """

    data = {"title": [],
            "author": [],
            "poem": []}
    
    total_num = list(range(num))
    with tqdm(total = len(total_num)) as pbar:
        d = 0   # 크롤링 성공하면 올라가는 num
        n = 0   # 성공여부에 상관없이 올라가는 num
        while d != num:
            # 주기적으로 sleep을 수행해 웹페이지가 죽지 않도록 구성
            if n % 50 == 0:
                time.sleep(random.uniform(1,3))
            
            current_id = lastest_id - n

            URL = f"https://poemlove.co.kr/bbs/board.php?bo_table=tb01&wr_id={current_id}"

            n += 1

            try:
                response = requests.get(URL)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "lxml")

                # 존재하지 않는 페이지로 접근할 경우, skip
                fail_message = soup.find("meta", {"name": 'title'})
                if "오류안내 페이지" in fail_message["content"]:
                    # tqdm.write(f"current_id: {current_id} is Fail.")
                    continue

                # tqdm.write(f"current_id: {current_id} is Success.")
                # 시 제목, 작가, 내용을 가져옴
                headline = soup.find("h1", {"itemprop": "headline"}).get_text().strip()
                author_box = soup.find_all("div", {"class": "panel panel-default view-head"})
                author = author_box[-1].find_all("strong")[0].get_text()          
                poem = soup.find("div", {'itemprop': 'description', 'class': 'view-content'}).get_text(separator="\n")

                # 시 내용에 대해 전처리 수행
                preprocessed_poem = preprocessing(headline, author, poem)

                # DataFrame 형식으로 저장
                data["title"].append(headline)
                data["author"].append(author)
                data["poem"].append(preprocessed_poem)
                
                # tqdm update
                pbar.update(1)
                d += 1


            except RequestException as e:
                print(f"Request Failed: {e}")

    # 후처리 후, csv파일로 저장
    df = pd.DataFrame(data)
    df = postprocessing(df)
    df.to_csv("./dataset/crawling_poem_100000.csv", index=False)


if __name__ == "__main__":
    crawling(lastest_id=271695, num=100000)
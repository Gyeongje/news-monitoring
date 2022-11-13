import random
import requests
from threading import Thread
from bs4 import BeautifulSoup
import time
import MySQLdb
import threading



headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}


def Connect_DB():
    conn = MySQLdb.connect(
        user="crawl_usr2",
        passwd="Test002",
        host="127.0.0.1",
        db="crawl_data2",
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor


def Crawling(agen, agen_num):
    conn, cursor = Connect_DB()
    cursor.execute("DROP TABLE IF EXISTS {}".format(agen))
    cursor.execute("CREATE TABLE {} (title text, url text, image text, time text)".format(agen))

    response = requests.get('https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=' + agen_num, headers=headers)
    if response.status_code == 200:
        # time.sleep(0.5)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        news_infos = soup.select('.photo')
        news_times = soup.select('.date.is_new') + soup.select('.date.is_outdated')

        idx = 5
        for i in range(0, idx):
            try:
                news_title = news_infos[i].img.get('alt')
                news_url = news_infos[i].a.get('href')
                news_image = news_infos[i].img.get('src')
                news_time = news_times[i].text

                for remove in '''.,'"“”‘’[]?!''':
                    news_title = news_title.replace(remove, '')
                for space in '·…':
                    news_title = news_title.replace(space, ' ')
            except:
                idx += 1
                continue

            '''
            print(news_title)
            print(news_url)
            print(news_image)
            print(news_time)
            print('***\n') '''

            cursor.execute(f"INSERT INTO {agen} VALUES(\"{news_title}\", \"{news_url}\", \"{news_image}\", \"{news_time}\")")
            conn.commit()
    conn.close()



 
if __name__ == '__main__':
    AGENCY = {
    "JTBC":"437", "KBS":"056", "MBC":"214", "SBS":"055", 
    "YTN":"052", "NEWS1":"421", "NEWSIS":"003", "연합뉴스":"001", 
    "조선일보":"023", "국민일보":"005", "경향신문":"032", "한겨레":"028"
    }

    
    threads = []
    while 1:
        start = time.perf_counter()    
        for A in AGENCY:
            t = threading.Thread(target=Crawling, args=(A, AGENCY[A], ))
            t.start()
            threads.append(t)
            
        for thread in threads:
            thread.join()

        
        finish = time.perf_counter()
        t = round(finish-start, 2)
        #print(f'Finished in {t} second(s)')
        try:
            time.sleep(10-t)
        except:
            time.sleep(5)
        
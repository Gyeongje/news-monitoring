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
        user="crawl_usr",
        passwd="Test001",
        host="127.0.0.1",
        db="crawl_data",
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor


def Crawling(agen, agen_num):
    conn, cursor = Connect_DB()
    cursor.execute("DROP TABLE IF EXISTS {}".format(agen))
    cursor.execute("CREATE TABLE {} (title text, url text, image text, view text)".format(agen))

    response = requests.get('https://media.naver.com/press/' + agen_num + '/ranking', headers=headers)

    if response.status_code == 200:
        # time.sleep(0.5)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        news_titles = soup.select('.list_title')  # soup.select('.list_title')[0]
        news_urls = soup.select('._es_pc_link')   # soup.select('._es_pc_link')[0].get('href')
        news_images = soup.select('.list_img')    # soup.select('.list_img')[0].img.get('src')
        news_views = soup.select('.list_view')    # soup.select('.list_view')[0].text.split()[1]

        idx = 5
        for i in range(0, idx):
            try:
                news_title = news_titles[i].text
                news_url = news_urls[i].get('href')
                news_image = news_images[i].img.get('src')
                if news_views != []:
                    news_view = news_views[i].text.split()[1] + '회'
                else:
                    news_view = "Not info.."

                for remove in '''.,'"“”‘’[]?!''':
                    news_title = news_title.replace(remove, '')
                for space in '·…':
                    news_title = news_title.replace(space, ' ')
            except:
                idx += 1
                continue

            '''print(f"INSERT INTO {agen} VALUES(\"{news_title}\", \"{news_url}\", \"{news_image}\", \"{news_view}\")")
            print(news_title)
            print(news_url)
            print(news_image)
            print(news_view)
            print('***\n')'''

            cursor.execute(f"INSERT INTO {agen} VALUES(\"{news_title}\", \"{news_url}\", \"{news_image}\", \"{news_view}\")")
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

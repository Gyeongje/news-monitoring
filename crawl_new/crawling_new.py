import random
import requests
from threading import Thread
from bs4 import BeautifulSoup
import time
import socket
import MySQLdb


headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
IP = ''
DB = ''


def Connect_DB(IP, DB):
    conn = MySQLdb.connect(
        user="root",
        passwd="test1234",
        host=IP,
        db=DB,
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor


def Crawling(agen, agen_num):
    conn, cursor = Connect_DB(IP, DB)

    response = requests.get('https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=' + agen_num, headers=headers)
    if response.status_code == 200:
        # time.sleep(0.5)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        news_infos = soup.select('.photo')
        news_times = soup.select('.date.is_new') + soup.select('.date.is_outdated')

        idx = 5
        news = []
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
                news.append([news_title, news_url, news_image, news_time])
            except:
                idx += 1
                continue
            #print(f"INSERT INTO {agen} VALUES(\"{news_title}\", \"{news_url}\", \"{news_image}\", \"{news_time}\")")
            #print(news)

        i = 1
        for n in news:
            cursor.execute(f"INSERT INTO {agen} VALUES (\"{i+1}\", \"{n[0]}\", \"{n[1]}\", \"{n[2]}\", \"{n[3]}\") \
                ON DUPLICATE KEY UPDATE title=\"{n[0]}\", url=\"{n[1]}\", image=\"{n[2]}\", time=\"{n[3]}\"")
            i+=1

        conn.commit()
        conn.close()



 
if __name__ == '__main__':
    AGENCY = {
    "JTBC":"437", "KBS":"056", "MBC":"214", "SBS":"055", 
    "YTN":"052", "NEWS1":"421", "NEWSIS":"003", "연합뉴스":"001", 
    "조선일보":"023", "국민일보":"005", "경향신문":"032", "한겨레":"028"
    }

    ip = socket.gethostbyname(socket.gethostname()).split('.')
    ip[-1] = str(3)
    ip = '.'.join(ip)
    IP = ip
    conn, cursor = Connect_DB(IP, DB)
    cursor.execute('CREATE DATABASE IF NOT EXISTS crawl_new DEFAULT CHARACTER SET utf8')
    conn.close()

    DB = 'crawl_new'
    conn, cursor = Connect_DB(IP, DB)
    for A in AGENCY:
        cursor.execute("CREATE TABLE IF NOT EXISTS {} (id int(1), title text, url text, image text, time text)".format(A))
        cursor.execute("ALTER TABLE {} ADD UNIQUE (id)".format(A))

    threads = []
    while 1:
        start = time.perf_counter()    
        for A in AGENCY:
            t = Thread(target=Crawling, args=(A, AGENCY[A], ))
            t.start()
            threads.append(t)
            
        for thread in threads:
            thread.join()

        
        finish = time.perf_counter()
        t = round(finish-start, 2)
        #print(f'Finished in {t} second(s)')
        time.sleep(5)
        
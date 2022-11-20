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
    while True:
        try:
            conn = MySQLdb.connect(
                user="root",
                passwd="test1234",
                host=IP,
                db=DB,
                charset="utf8"
            )
            break
        except:
            time.sleep(5)
    cursor = conn.cursor()
    return conn, cursor


def Crawling(agen, agen_num):
    conn, cursor = Connect_DB(IP, DB)
    response = requests.get('https://media.naver.com/press/' + agen_num + '/ranking', headers=headers)

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        news_titles = soup.select('.list_title')  # soup.select('.list_title')[0]
        news_urls = soup.select('._es_pc_link')   # soup.select('._es_pc_link')[0].get('href')
        news_images = soup.select('.list_img')    # soup.select('.list_img')[0].img.get('src')
        news_views = soup.select('.list_view')    # soup.select('.list_view')[0].text.split()[1]

        idx = 5
        news = []
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
                news.append([news_title, news_url, news_image, news_view])
            except:
                idx += 1
                continue

        i = 1
        for n in news:
            cursor.execute(f"INSERT INTO {agen} VALUES (\"{i+1}\", \"{n[0]}\", \"{n[1]}\", \"{n[2]}\", \"{n[3]}\") \
                ON DUPLICATE KEY UPDATE title=\"{n[0]}\", url=\"{n[1]}\", image=\"{n[2]}\", view=\"{n[3]}\"")
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
    ip[-1] = str(2)
    ip = '.'.join(ip)
    IP = ip
    conn, cursor = Connect_DB(IP, DB)
    cursor.execute('CREATE DATABASE IF NOT EXISTS crawl_main DEFAULT CHARACTER SET utf8')
    conn.close()

    DB = 'crawl_main'
    conn, cursor = Connect_DB(IP, DB)
    for A in AGENCY:
        cursor.execute("CREATE TABLE IF NOT EXISTS {} (id int(1), title text, url text, image text, view text)".format(A))
        cursor.execute("ALTER TABLE {} ADD UNIQUE (id)".format(A))
    conn.close()

    while 1:
        threads = []
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

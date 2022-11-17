from textrank import *
from konlpy.tag import Komoran
import random
import requests
from threading import Thread
from bs4 import BeautifulSoup
import time
import MySQLdb
import threading
import socket


headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}


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


def extract_ip(n):
    ip = socket.gethostbyname(socket.gethostname()).split('.')
    ip[-1] = str(n)
    ip = '.'.join(ip)
    return ip

komoran = Komoran()
def komoran_tokenizer(sent):
        words = komoran.pos(sent, join=True)
        words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words


if __name__ == '__main__':
    AGENCY = {
    "JTBC":"437", "KBS":"056", "MBC":"214", "SBS":"055", 
    "YTN":"052", "NEWS1":"421", "NEWSIS":"003", "연합뉴스":"001", 
    "조선일보":"023", "국민일보":"005", "경향신문":"032", "한겨레":"028"
    }

    crawl_main_ip = extract_ip(4)
    crawl_new_ip = extract_ip(3)
    crawl_keyword_ip = extract_ip(2)
    while 1:
        start = time.perf_counter()
        conn, cursor = Connect_DB(crawl_main_ip,'crawl_main')
        conn2, cursor2 = Connect_DB(crawl_new_ip,'crawl_new')
        result = ()

        for A in AGENCY:
            while 1:
                try:
                    cursor.execute(f"select title from {A}")
                    cursor2.execute(f"select title from {A}")
                    break
                except:
                    continue
            result += (cursor.fetchall() + cursor2.fetchall())
        conn2.close()
        conn.close()
        
        sents = []
        for i in result:
            sents.append(i[0])
        #print(sents)
        
        summarizer = KeysentenceSummarizer(
            tokenize = komoran_tokenizer,
            min_sim = 0.3,
            verbose = False
        )
        summarizer = KeywordSummarizer(tokenize=komoran_tokenizer, min_count=2, min_cooccurrence=1)
        keywords = summarizer.summarize(sents, topk=30)
        rank = []
        count = 1
        for k in keywords:
            word = k[0].split('/')
            if word[1] == "NNP" or word[1] == "NNG":
                rank.append([word[0], round(k[1], 6)])
                if count >= 20:
                    break
                count += 1
        #print(rank)
        conn, cursor = Connect_DB(crawl_keyword_ip, '')
        cursor.execute('CREATE DATABASE IF NOT EXISTS crawl_keyword DEFAULT CHARACTER SET utf8')
        conn.close()

        conn, cursor = Connect_DB(crawl_keyword_ip,'crawl_keyword')

        cursor.execute("CREATE TABLE IF NOT EXISTS KEYWORDS (id int(2), keyword text, persent text)")
        cursor.execute("ALTER TABLE KEYWORDS ADD UNIQUE (id)")

        i=0
        for r in rank:
            cursor.execute(f"INSERT INTO KEYWORDS VALUES (\"{i+1}\", \"{r[0]}\", \"{r[1]}\") \
                ON DUPLICATE KEY UPDATE keyword=\"{r[0]}\", persent=\"{r[1]}\"")
            i+=1
        conn.commit()
        conn.close()
        
        finish = time.perf_counter()
        t = round(finish-start, 2)
        #print(f'Finished in {t} second(s)')
        time.sleep(5)
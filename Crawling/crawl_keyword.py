from textrank import *
from konlpy.tag import Komoran
import random
import requests
from threading import Thread
from bs4 import BeautifulSoup
import time
import MySQLdb
import threading


def Connect_DB(n, p):
    conn = MySQLdb.connect(
        user="crawl_usr"+n,
        passwd="Test00"+p,
        host="127.0.0.1",
        db="crawl_data"+n,
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor


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
    while 1:
        start = time.perf_counter()
        conn, cursor = Connect_DB('','1')
        conn2, cursor2 = Connect_DB('2','2')
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
        conn, cursor = Connect_DB('3','3')
        cursor.execute("DROP TABLE IF EXISTS KEYWORDS")
        cursor.execute("CREATE TABLE KEYWORDS (keyword text, persent text)")
        for r in rank:
            cursor.execute(f"INSERT INTO KEYWORDS VALUES(\"{r[0]}\", \"{r[1]}\")")
        conn.commit()
        conn.close()
        
        finish = time.perf_counter()
        t = round(finish-start, 2)
        try:
            time.sleep(10-t)
        except:
            time.sleep(5)
# 실시간 뉴스 모니터링 사이트
2022 소프트웨어 페스티벌 데이터 분석 부문 실시간 뉴스 모니터링 서비스입니다.

> 개발 언어: HTML, CSS, JS, PHP, Python <br>
> 활용 플랫폼: Docker, Ubuntu, MYSQL, APACHE, Naver NEWs Template <br>
> 활용 라이브러리: TextRank, requests, beautifulsoup4, MySQLdb, scikit-learn, konlpy
<br>

+ 실시간 뉴스 모니터링 서비스는 최근 주요 이슈를 보여주는 실시간 뉴스 모니터링 시스템입니다.
+ 최근 카카오, 이태원 사태 등과 같은 최근 주요 이슈를 뉴스 데이터를 통해 신속하게 알 수 있도록 하는 것이 목적입니다.
+ 언론사 중 방송/통신사 8곳, 종합지 4곳을 대상으로 최신 기사 정보 + 주요 기사 내용을 획득하여 <br>
웹사이트를 통해 조회수에 따른 **주요 뉴스**, 시간에 따른 **최근 뉴스**를 보여줍니다.
+ 또한 각 기사 제목을 TextRank Library를 활용하여 키워드 및 언급도를 분석하고 키워드 언급 순위를 추출하여 **실시간 검색어 순위**를 구현하였습니다. <br>

## 세부구현
## 주요 뉴스 기사 (crawl_main)
> 개발 환경: alpine linux(Docker), Python 3.8 <br>
> 활용 플랫폼: Docker, Mysql <br>
> 매체 선정: JTBC, KBS, MBC, SBS, YTN, NEWS1, NEWSIS, 연합뉴스, 조선일보, 국민일보, 경향신문, 한겨레 (방송/통신사 8, 종합지 4) <br>

![image](https://user-images.githubusercontent.com/31283542/202893364-2c455631-b1fb-49fd-8ee0-ba2cd8071de9.png)
+ NAVER 플랫폼을 통해 각 방송사의 많이 시청한 기사 목록을 확인할 수 있음. 
+ 해당 부분을 이용하여 위의 12곳에 해당하는 매체의 정보를 Crawling.
<br>

### Dockerfile 
```
FROM python:3.8-alpine

RUN apk add --no-cache mariadb-dev build-base

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade requests \
    && python3 -m pip install --upgrade beautifulsoup4 \
    && python3 -m pip install --upgrade mysqlclient

RUN mkdir /home/crawl_main
RUN cd /home/crawl_main

WORKDIR /home/crawl_main
COPY crawling_main.py .

CMD ["python", "crawling_main.py"]
```
+ resource가 많이 필요치 않으므로 alpine linux + python3.8 image를 Pull.
+ Moudle import를 위해 필요 libray를 빌드 (build-base(gcc,g++...), mariadb-dev)
+ HTTP 통신을 위한 requests, 파싱을 위한 beautifulsoup4, MYSQL DB 접속을 위해 mysqlclient를 설치 

### Source (crawling_main.py)
``` python
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
```
+ IP, Database Name을 전달받아, 특정 MySQL DB에 접속 후 통신할 수 있는 conn, cursor를 return. 

``` python
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
```
+ 각 방송사의 많이 시청한 기사 목록에 해당하는 HTML Source를 requests.get을 통해 DATA를 받음
+ 기사 제목, 기사 link, 기사 image, 기사 조회수를 BeautifulSoup를 통해 파싱함.
+ 5위까지의 data를 배열을 통해 저장시킨뒤, 연결했던 MySQL Connection을 통해 DATA를 Table에 삽입.
    ``` 
    cursor.execute(f"INSERT INTO {agen} VALUES (\"{i+1}\", \"{n[0]}\", \"{n[1]}\", \"{n[2]}\", \"{n[3]}\") \
                ON DUPLICATE KEY UPDATE title=\"{n[0]}\", url=\"{n[1]}\", image=\"{n[2]}\", view=\"{n[3]}\"") 
    ```
    + 해당 QUERY는 Data를 TABLE에 insert할 때 만약 table에 data가 존재하지 않으면 Insert, 존재하지 않다면 UPDATE하는 방식을 사용.
    + 만약 table을 삭제한 뒤 insert하는 쿼리를 사용하면 site에서 해당 DB를 접속했을 때 데이터가 없는 경우 Error가 발생하기 때문.

``` python
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
```
+ MYSQL Connection 후 Database, Table을 생성. 
+ 각 1~5순위에 해당하는 data를 구분하고 UPDATE 방식을 사용하기 위해 TABLE에 UNIQUE id ALTER함.
+ 추후 각각의 매체의 기사를 차례차례 가져오는 방식은 비효율적이기에 Multi Thread 방식을 사용하여 기사 수집 시간을 단축.   

## 최신 뉴스 기사 (crawl_new)
> 개발 환경: alpine linux(Docker), Python 3.8 <br>
> 활용 플랫폼: Docker, Mysql <br>
> 매체 선정: JTBC, KBS, MBC, SBS, YTN, NEWS1, NEWSIS, 연합뉴스, 조선일보, 국민일보, 경향신문, 한겨레 (방송/통신사 8, 종합지 4) <br>

![image](https://user-images.githubusercontent.com/31283542/202894259-05fbb123-e41b-47bc-83c8-af4e4d609deb.png)
+ NAVER 플랫폼을 통해 각 방송사의 최신 기사 목록을 확인할 수 있음. 
+ 해당 부분을 이용하여 위의 12곳에 해당하는 매체의 정보를 Crawling.

### Dockerfile 
```
FROM python:3.8-alpine

RUN apk add --no-cache mariadb-dev build-base

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade requests \
    && python3 -m pip install --upgrade beautifulsoup4 \
    && python3 -m pip install --upgrade mysqlclient

RUN mkdir /home/crawl_new
RUN cd /home/crawl_new

WORKDIR /home/crawl_new
COPY crawling_new.py .

CMD ["python", "crawling_new.py"]
```

### Source (crawling_new.py)
``` python
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

        i = 1
        for n in news:
            cursor.execute(f"INSERT INTO {agen} VALUES (\"{i+1}\", \"{n[0]}\", \"{n[1]}\", \"{n[2]}\", \"{n[3]}\") \
                ON DUPLICATE KEY UPDATE title=\"{n[0]}\", url=\"{n[1]}\", image=\"{n[2]}\", time=\"{n[3]}\"")
            i+=1

        conn.commit()
    conn.close()
```
+ 위 crawling_main 부분과 대부분 동일. 
+ 최신 기사에 해당하는 크롤링 부분만 다름.
    + 기사 제목, 기사 link, 기사 image, 기사 업로드 time 파싱.



## 주요-최신 뉴스 기사 검색어 순위 (crawl_keyword)


## 사이트 구축 (crawl_site)

## 서비스 관리

<br><br>
## Architecture
![image](https://user-images.githubusercontent.com/31283542/202607758-82e9e9cf-88c0-497d-bb0b-bda295d1d3a1.png)
<br><br>
## DEMO 영상
[Youtube] https://youtu.be/7l4j4UEoYko
<img src="https://user-images.githubusercontent.com/31283542/201531091-325cd852-a1ea-4488-ba7e-e1ed36dd6de3.gif" width="1080" height="900"/>

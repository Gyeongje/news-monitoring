FROM frolvlad/alpine-python-machinelearning

ENV JAVA_HOME /usr/lib/jvm/java-1.7-openjdk/jre
ENV LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk/lib/server/
RUN apk upgrade --no-cache \
  && apk update \
  && apk add --no-cache \
  mariadb-dev \
  build-base \
  python3-dev \
  openjdk11


RUN python -m pip install --upgrade pip \
    && python -m pip install --upgrade requests \
    && python -m pip install --upgrade beautifulsoup4 \
    && python -m pip install --upgrade mysqlclient \
    && python -m pip install --upgrade konlpy


RUN mkdir /home/crawl_keyword \ 
    && cd /home/crawl_keyword \
    && wget https://github.com//Gyeongje/news-monitoring/raw/main/Crawling/textrank.tar \
    && tar -xvf /home/crawl_keyword/textrank.tar \
    && rm textrank.tar && cd textrank \
    && python3 /home/crawl_keyword/textrank/setup.py build

COPY crawl_keyword.py /home/crawl_keyword/textrank

CMD ["python", "/home/crawl_keyword/textrank/crawl_keyword.py"]
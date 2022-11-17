FROM python:3.8-alpine

RUN apk add --no-cache mariadb-dev build-base

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade requests \
    && python3 -m pip install --upgrade beautifulsoup4 \
    && python3 -m pip install --upgrade mysqlclient

RUN mkdir /home/crawl_new
RUN cd /home/crawl_new

COPY crawling_new.py /home/crawl_new

CMD ["python", "/home/crawl_new/crawling_new.py"]
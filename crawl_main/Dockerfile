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
#CMD ["/bin/sh"]
FROM oberd/php-8.0-apache

RUN apt-get update && apt-get install -y vim 
RUN echo "AddType application/x-httpd-php .html" >> /etc/apache2/mods-enabled/mime.conf

COPY src/ .
RUN cp crawling_main.html index.html
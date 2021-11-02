# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y default-libmysqlclient-dev gcc
RUN pip3 install mysqlclient==2.0.3
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
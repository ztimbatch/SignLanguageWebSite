FROM python:3.10-slim-buster
LABEL authors="ztimbatch"

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip --no-cache-dir install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

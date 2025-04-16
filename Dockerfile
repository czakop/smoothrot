FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /smoothrot

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /smoothrot

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /smoothrot

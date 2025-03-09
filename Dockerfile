FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /smoothrot

COPY requirements.txt /smoothrot

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /smoothrot

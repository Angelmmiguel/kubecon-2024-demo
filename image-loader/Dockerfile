FROM bitnami/python:3.10.14

COPY model-downloader.py /app

RUN pip install oras
RUN mkdir /llm/

CMD ["python", "/app/model-downloader.py"]

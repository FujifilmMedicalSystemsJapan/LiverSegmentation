FROM nvcr.io/nvidia/tensorflow:19.07-py3

COPY . /app
WORKDIR /app

CMD ["python", "main.py"]

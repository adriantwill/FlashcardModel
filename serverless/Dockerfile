FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY serverless/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY serverless/handler.py ./handler.py
COPY serverless/adapter ./adapter

CMD ["python", "-u", "handler.py"]

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install \
    --no-cache-dir \
    --break-system-packages \
    --ignore-installed cryptography \
    runpod==1.12.0

RUN python -m pip install \
    --no-cache-dir \
    --break-system-packages \
    -r requirements.txt

COPY handler.py ./handler.py
COPY adapter ./adapter

CMD ["python", "-u", "handler.py"]

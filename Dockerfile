FROM python:3.11-slim

EXPOSE 7860

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

WORKDIR /app

COPY . .

RUN python3 -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir six packaging python-dateutil platformdirs pywin32 && \
    pip install --no-cache-dir -r requirements.txt

VOLUME [ "/app/assets" ]
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python3", "main/app/app.py"]

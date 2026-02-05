FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]

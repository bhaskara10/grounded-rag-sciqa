FROM python:3.11-slim
WORKDIR /app

COPY libs/schema /app/libs/schema
COPY libs/events /app/libs/events
COPY libs/common /app/libs/common
COPY services/eval /app/services/eval

RUN pip install --no-cache-dir \
    /app/libs/schema \
    /app/libs/events \
    /app/libs/common \
    /app/services/eval

EXPOSE 8003
CMD ["uvicorn", "services.eval.app.main:app", "--host", "0.0.0.0", "--port", "8003"]

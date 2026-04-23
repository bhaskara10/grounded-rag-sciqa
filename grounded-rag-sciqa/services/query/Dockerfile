FROM python:3.11-slim
WORKDIR /app

COPY libs/schema /app/libs/schema
COPY libs/events /app/libs/events
COPY libs/common /app/libs/common
COPY services/query /app/services/query

RUN pip install --no-cache-dir \
    /app/libs/schema \
    /app/libs/events \
    /app/libs/common \
    /app/services/query

EXPOSE 8002
CMD ["uvicorn", "services.query.app.main:app", "--host", "0.0.0.0", "--port", "8002"]

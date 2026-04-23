FROM python:3.11-slim AS base
WORKDIR /app

COPY libs/schema /app/libs/schema
COPY libs/events /app/libs/events
COPY libs/common /app/libs/common
COPY services/ingest /app/services/ingest

RUN pip install --no-cache-dir \
    /app/libs/schema \
    /app/libs/events \
    /app/libs/common \
    /app/services/ingest

EXPOSE 8001
CMD ["uvicorn", "services.ingest.app.main:app", "--host", "0.0.0.0", "--port", "8001"]

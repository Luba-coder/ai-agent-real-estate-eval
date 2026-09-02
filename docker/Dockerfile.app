FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    beautifulsoup4 \
    pydantic \
    langchain==0.2.16 \
    langchain-core==0.2.41 \
    langchain-community==0.2.17 \
    langchain-openai==0.1.25 \
    langchain-ollama \
    langsmith \
    python-dotenv \
    tiktoken \
    langfuse

COPY app /app/app

EXPOSE 8004

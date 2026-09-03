FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config.example.yaml /app/config.yaml
ENV INFINITUM_CONFIG=/app/config.yaml
EXPOSE 8788
CMD ["infinitum", "serve", "--config", "/app/config.yaml"]

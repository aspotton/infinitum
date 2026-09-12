# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home infinitum
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    INFINITUM_CONFIG=/config/config.yaml
EXPOSE 8788
# ponytail: port hardcoded to default server.port=8788; parse cfg under INFINITUM_CONFIG if that ever must adapt
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8788/health', timeout=4).status==200 else 1)"
USER 10001
CMD ["infinitum", "serve"]

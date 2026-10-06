FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.lock pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.lock \
    && /opt/venv/bin/pip install --no-deps .

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AGENT_DB_PATH=/data/conversations.sqlite AGENT_MODEL_BACKEND=stub AGENT_STAFF_PANEL=false
RUN groupadd --gid 10001 agent && useradd --uid 10001 --gid agent --no-create-home agent \
    && mkdir /data && chown agent:agent /data
COPY --from=build /opt/venv /opt/venv
USER 10001:10001
WORKDIR /app
EXPOSE 5000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz',timeout=2)"
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "60", "--access-logfile", "-", "--logger-class", "isp_support_agent.server.JsonGunicornLogger", "--access-logformat", "{\"event\":\"http_request\",\"method\":\"%(m)s\",\"status\":%(s)s,\"duration_us\":%(D)s}", "isp_support_agent.server:create_app()"]

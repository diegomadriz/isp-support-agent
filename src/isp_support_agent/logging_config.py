import json
import logging


class JsonEventFormatter(logging.Formatter):
    """Preserve our fixed JSON events; include third-party logger and exception type, without secret-bearing message text."""

    def format(self, record: logging.LogRecord) -> str:
        if record.name in {"isp_support_agent", "gunicorn.access"}:
            try:
                event = json.loads(record.getMessage())
                if isinstance(event, dict):
                    return json.dumps(event, sort_keys=True)
            except (ValueError, TypeError):
                pass
        return json.dumps(
            {
                "event": "server_log",
                "logger": record.name,
                "level": record.levelname,
                "exception_type": record.exc_info[0].__name__ if record.exc_info else None,
            },
            sort_keys=True,
        )


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonEventFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

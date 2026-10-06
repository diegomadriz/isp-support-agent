"""Single-worker production WSGI entry point with durable conversation storage."""

import atexit

from gunicorn.glogging import Logger
from pydantic import ValidationError

from .logging_config import JsonEventFormatter, configure_logging
from .runtime import AgentRuntime
from .settings import Settings
from .web import create_app as web_app


def create_app():
    configure_logging()
    try:
        settings = Settings.from_env()
    except ValidationError:
        raise RuntimeError("Invalid agent configuration; check environment settings") from None
    runtime = AgentRuntime.persistent(settings)
    app = web_app(runtime)

    def close():
        app.extensions["background"].close()
        sender = app.extensions["message_sender"]
        if hasattr(sender, "close"):
            sender.close()
        runtime.close()

    atexit.register(close)
    return app


class JsonGunicornLogger(Logger):
    def setup(self, cfg):
        super().setup(cfg)
        for logger in (self.error_log, self.access_log):
            for handler in logger.handlers:
                handler.setFormatter(JsonEventFormatter())

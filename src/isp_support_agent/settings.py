import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_backend: Literal["stub", "ollama", "openai"] = "stub"
    model_base_url: str = ""
    model_name: str = ""
    model_key: str = Field(default="", repr=False)
    rewrite: bool | None = None
    rewrite_timeout: float = Field(default=8, gt=0, le=8)
    company_name: str = "Soporte"
    bot_name: str = ""
    host: str = "127.0.0.1"
    session_secret: str = Field(default="", repr=False)
    staff_panel: bool = False
    whatsapp_sender: Literal["fake", "cloud"] = "fake"
    whatsapp_base_url: str = ""
    whatsapp_access_token: str = Field(default="", repr=False)
    scenario: str = "healthy"
    db_path: Path = Path("data/conversations.sqlite")
    port: int = Field(default=5000, ge=1, le=65535)
    probe_timeout: float = Field(default=0.1, gt=0, le=30)
    probe_latency: float = Field(default=0.002, ge=0, le=30)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    model_timeout: float = Field(default=6, gt=0, le=6)
    lockout_seconds: int = Field(default=900, ge=1)
    webhook_max_age: int = Field(default=600, ge=1)
    rate_limit: int = Field(default=30, ge=1)
    whatsapp_verify_token: str = Field(default="", repr=False)
    whatsapp_app_secret: str = Field(default="", repr=False)

    @field_validator("rewrite", mode="before")
    @classmethod
    def automatic_rewrite(cls, value):
        return None if value in ("", "auto") else value

    @field_validator("company_name", "bot_name")
    @classmethod
    def brand_text(cls, value):
        import re

        value = value.strip()
        if len(value) > 64 or (value and not re.fullmatch(r"[\w ÁÉÍÓÚÜÑáéíóúüñ&'-]+", value)):
            raise ValueError("Use a short plain-text brand name")
        return value

    @model_validator(mode="after")
    def validate_model(self):
        if self.model_backend != "stub":
            if not self.model_name or not self.model_base_url.startswith(("http://", "https://")):
                raise ValueError("Configure AGENT_MODEL_BASE_URL and AGENT_MODEL_NAME")
            if self.model_backend == "openai" and not self.model_key:
                raise ValueError("Configure AGENT_MODEL_KEY")
        if self.rewrite is None:
            object.__setattr__(self, "rewrite", self.model_backend != "stub")
        if self.whatsapp_sender == "cloud" and (
            not self.whatsapp_access_token or not self.whatsapp_base_url.startswith("https://")
        ):
            raise ValueError(
                "Configure AGENT_WHATSAPP_ACCESS_TOKEN and HTTPS AGENT_WHATSAPP_BASE_URL"
            )
        return self

    @classmethod
    def from_env(cls):
        return cls.model_validate(
            {
                name: os.environ[f"AGENT_{name.upper()}"]
                for name in cls.model_fields
                if f"AGENT_{name.upper()}" in os.environ
            }
        )

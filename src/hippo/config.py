from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    hippo_model: str = "gpt-4o-mini"
    hippo_demo_model: str = "gpt-4o"
    # auto: Seahorse when SEAHORSE_API_KEY is set, else Chroma. Or force seahorse|chroma.
    hippo_memory_backend: str = "auto"
    seahorse_api_key: str = ""
    seahorse_table_episodes: str = "hippo_episodes"
    seahorse_table_facts: str = "hippo_facts"
    hippo_token_budget: int = 8000
    hippo_data_dir: Path = Path(".hippo")
    hippo_workspace: Path = Path(".")


def load_settings() -> Settings:
    s = Settings()
    s.hippo_data_dir.mkdir(parents=True, exist_ok=True)
    return s

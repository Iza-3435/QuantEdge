"""Configuration management with environment variable support."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"


class PatternRetrievalConfig(BaseModel):
    embedding_dim: int = 128
    top_k: int = 10
    window_size: int = 20
    forward_periods: int = 5
    index_type: str = "HNSW"
    hnsw_m: int = 32
    hnsw_ef_search: int = 128
    hnsw_ef_construction: int = 200


class MarketCLIPConfig(BaseModel):
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 256
    price_input_dim: int = 10
    learning_rate: float = 0.0001
    batch_size: int = 32
    temperature: float = 0.07


class EnsembleConfig(BaseModel):
    n_models: int = 5
    voting_strategy: str = "weighted"


class ModelConfig(BaseModel):
    pattern_retrieval: PatternRetrievalConfig = PatternRetrievalConfig()
    market_clip: MarketCLIPConfig = MarketCLIPConfig()
    ensemble: EnsembleConfig = EnsembleConfig()


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "market-intelligence"
    artifact_location: str = "./mlruns"
    model_registry: bool = True


class PrometheusConfig(BaseModel):
    enabled: bool = True
    port: int = 9090
    metrics_path: str = "/metrics"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/app.log"
    rotation: str = "500 MB"
    retention: str = "30 days"


class MonitoringConfig(BaseModel):
    prometheus: PrometheusConfig = PrometheusConfig()
    logging: LoggingConfig = LoggingConfig()


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    cache_ttl: int = 3600
    max_connections: int = 50


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000


class AuthConfig(BaseModel):
    enabled: bool = True
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


class CORSConfig(BaseModel):
    enabled: bool = True
    origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]


class APIConfig(BaseModel):
    rate_limit: RateLimitConfig = RateLimitConfig()
    auth: AuthConfig = AuthConfig()
    cors: CORSConfig = CORSConfig()


class NewsConfig(BaseModel):
    alpha_vantage_key: Optional[str] = None
    yahoo_enabled: bool = True
    cache_enabled: bool = True
    cache_dir: str = "data/cache/news"


class PatternsConfig(BaseModel):
    save_dir: str = "data/processed/patterns"
    backup_enabled: bool = True
    backup_interval_hours: int = 24


class DataConfig(BaseModel):
    news: NewsConfig = NewsConfig()
    patterns: PatternsConfig = PatternsConfig()


class InferenceConfig(BaseModel):
    batch_size: int = 32
    timeout_seconds: int = 30
    uncertainty_quantification: bool = True
    min_confidence_threshold: float = 0.5


class DriftDetectionConfig(BaseModel):
    enabled: bool = True
    window_size: int = 1000
    check_interval_hours: int = 24
    threshold_psi: float = 0.1
    threshold_js: float = 0.1
    retrain_on_drift: bool = True


class Settings(BaseSettings):
    """Main settings class with environment variable support."""

    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLflowConfig = MLflowConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    data: DataConfig = DataConfig()
    inference: InferenceConfig = InferenceConfig()
    drift_detection: DriftDetectionConfig = DriftDetectionConfig()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    config_file = Path(config_path)

    if not config_file.exists():
        return {}

    with open(config_file, 'r') as f:
        config_str = f.read()

    # Substitute environment variables
    for match in ["${JWT_SECRET}", "${ALPHA_VANTAGE_KEY}"]:
        env_var = match.strip("${}")
        value = os.getenv(env_var, "")
        config_str = config_str.replace(match, value)

    return yaml.safe_load(config_str)


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get cached settings instance."""

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")

    # Load from YAML if exists
    yaml_config = load_config_from_yaml(config_path)

    if yaml_config:
        return Settings(**yaml_config)

    # Fall back to environment variables and defaults
    return Settings()


# Global settings instance
settings = get_settings()

"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable

# API Metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Model Metrics
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0)
)

model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_type", "status"]
)

model_confidence_score = Histogram(
    "model_confidence_score",
    "Model confidence score distribution",
    ["model_type"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Pattern Retrieval Metrics
pattern_retrieval_count = Counter(
    "pattern_retrieval_count",
    "Number of pattern retrievals",
    ["top_k"]
)

pattern_similarity_score = Histogram(
    "pattern_similarity_score",
    "Pattern similarity score distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Cache Metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"]
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"]
)

# System Metrics
model_loaded = Gauge(
    "model_loaded",
    "Whether model is loaded (1) or not (0)",
    ["model_type"]
)

pattern_database_size = Gauge(
    "pattern_database_size",
    "Number of patterns in database"
)

# Drift Detection Metrics
drift_detected = Counter(
    "drift_detected_total",
    "Number of times drift was detected",
    ["metric_type"]
)

drift_score = Gauge(
    "drift_score",
    "Current drift score",
    ["metric_type"]
)

# Application Info
app_info = Info("app_info", "Application information")


def track_prediction_time(model_type: str):
    """Decorator to track model prediction time."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                model_inference_duration_seconds.labels(model_type=model_type).observe(duration)
                model_predictions_total.labels(model_type=model_type, status="success").inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                model_inference_duration_seconds.labels(model_type=model_type).observe(duration)
                model_predictions_total.labels(model_type=model_type, status="error").inc()
                raise
        return wrapper
    return decorator


def track_api_request(endpoint: str, method: str):
    """Decorator to track API request metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
                api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        return wrapper
    return decorator


def track_cache_access(cache_type: str, hit: bool):
    """Track cache hit/miss."""
    if hit:
        cache_hits_total.labels(cache_type=cache_type).inc()
    else:
        cache_misses_total.labels(cache_type=cache_type).inc()


def set_model_status(model_type: str, loaded: bool):
    """Set model loaded status."""
    model_loaded.labels(model_type=model_type).set(1 if loaded else 0)

import os
import secrets
import logging
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)

DEFAULT_ADMIN_KEY = "fruitvision-admin-dev-key-change-in-prod"


def get_expected_api_key() -> str:
    """Retrieve expected admin key from environment or fallback to dev key with a warning."""
    env_key = (
        os.getenv("FRUITVISION_API_KEY")
        or os.getenv("FRUITVISION_ADMIN_KEY")
        or os.getenv("ADMIN_API_KEY")
    )
    if env_key:
        return env_key

    is_production = os.getenv("ENVIRONMENT", "").lower() in ["production", "prod"]
    if is_production:
        logger.critical("[SECURITY] Running in production without FRUITVISION_ADMIN_KEY configured! Denying admin access.")
        return ""

    logger.warning("[SECURITY] Admin API key is using development default secret. Set FRUITVISION_ADMIN_KEY in production.")
    return DEFAULT_ADMIN_KEY


def verify_admin_key(
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
    x_admin_key: Optional[str] = Security(ADMIN_KEY_HEADER),
) -> str:
    """
    Enforce API key authentication for administrative and destructive operations
    (e.g., purging inspection database, deleting records, reloading model weights).
    Accepts 'X-API-Key' or 'X-Admin-Key'.
    Returns 401 if missing, 403 if invalid.
    """
    provided_key = x_api_key or x_admin_key
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing administrative API key. Provide 'X-API-Key' or 'X-Admin-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    expected_key = get_expected_api_key()
    if not expected_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid administrative API key."
        )

    return provided_key

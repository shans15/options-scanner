from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAuthError(RuntimeError):
    """Raised when Kalshi auth env vars are missing or the private key cannot be loaded."""


class KalshiSigner:
    """Sign Kalshi API requests with RSA-PSS.

    Reads KALSHI_ACCESS_KEY_ID and KALSHI_PRIVATE_KEY_PATH from environment.
    Caches the loaded private key in memory.
    """

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
    ) -> None:
        """If args are None, reads from env vars KALSHI_ACCESS_KEY_ID and KALSHI_PRIVATE_KEY_PATH.

        Raises KalshiAuthError if neither args nor env are set, or key file is missing/invalid.
        """
        self.access_key_id = access_key_id or os.environ.get("KALSHI_ACCESS_KEY_ID")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")

        if not self.access_key_id or self.access_key_id == "PASTE_YOUR_ACCESS_KEY_ID_HERE":
            raise KalshiAuthError(
                "KALSHI_ACCESS_KEY_ID not set. Add to .env or pass explicitly. "
                "Sign up at kalshi.com → Settings → API Keys to get one."
            )
        if not self.private_key_path:
            raise KalshiAuthError("KALSHI_PRIVATE_KEY_PATH not set.")
        if not Path(self.private_key_path).exists():
            raise KalshiAuthError(f"Private key file not found: {self.private_key_path}")

        try:
            with open(self.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        except Exception as e:
            raise KalshiAuthError(
                f"Could not load private key from {self.private_key_path}: {e}"
            )

    def sign_headers(self, method: str, path: str) -> dict[str, str]:
        """Return the three Kalshi auth headers for a request.

        path: the URL path starting with '/' (no scheme/host). Include query string if any.
        """
        ts_ms = str(int(time.time() * 1000))
        message = f"{ts_ms}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(signature).decode("ascii")
        return {
            "KALSHI-ACCESS-KEY": self.access_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
        }

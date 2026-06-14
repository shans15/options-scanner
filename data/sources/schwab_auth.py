"""Schwab OAuth 2.0 authorization-code flow.

Rolls our own client since `schwab-py` requires Python 3.10+ and we're on 3.9.

Reads from env:
  SCHWAB_CLIENT_ID
  SCHWAB_CLIENT_SECRET
  SCHWAB_CALLBACK_URL    (must match what was registered at developer.schwab.com)
  SCHWAB_TOKEN_PATH      (where to persist the refresh token, ~/.config/schwab/token.json)

Token lifecycle:
  access_token:  valid 30 minutes, auto-refreshed on demand
  refresh_token: valid 7 days, requires manual re-auth via browser when expired
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests


_AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"
_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

logger = logging.getLogger(__name__)


class SchwabAuthError(RuntimeError):
    """Raised when Schwab auth env vars are missing, or token exchange/refresh fails."""


class SchwabAuth:
    """Manages the OAuth token lifecycle for Schwab API requests.

    Usage:
        auth = SchwabAuth()              # reads env, loads token if exists
        token = auth.get_access_token()  # returns fresh token, refreshing if needed
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        token_path: Optional[str] = None,
    ) -> None:
        self.client_id = self._resolve(client_id, "SCHWAB_CLIENT_ID")
        self.client_secret = self._resolve(client_secret, "SCHWAB_CLIENT_SECRET")
        self.callback_url = self._resolve(callback_url, "SCHWAB_CALLBACK_URL", allow_file=False)
        token_path_str = token_path or os.environ.get("SCHWAB_TOKEN_PATH")

        for name, val in [
            ("SCHWAB_CLIENT_ID", self.client_id),
            ("SCHWAB_CLIENT_SECRET", self.client_secret),
            ("SCHWAB_CALLBACK_URL", self.callback_url),
            ("SCHWAB_TOKEN_PATH", token_path_str),
        ]:
            if not val or val.startswith("PASTE_") or val.startswith("YOUR_"):
                raise SchwabAuthError(
                    f"{name} not set in environment (got {val!r}). "
                    f"Add it to .env and reload the shell."
                )

        self.token_path = Path(token_path_str)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

        self._token_data: Optional[dict] = None
        self._load_token_if_exists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(passed: Optional[str], env_name: str, allow_file: bool = True) -> Optional[str]:
        """Return the credential string. If the env value points to an existing file,
        read the file's content (stripped of whitespace) — supports keeping secrets
        in chmod-600 files outside .env."""
        raw = passed or os.environ.get(env_name)
        if not raw:
            return raw
        if allow_file and len(raw) < 512 and Path(raw).is_file():
            try:
                return Path(raw).read_text().strip()
            except OSError:
                return raw
        return raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_access_token(self) -> str:
        """Return a fresh access token, refreshing or initiating OAuth if needed."""
        if self._token_data is None:
            raise SchwabAuthError(
                "No saved token. Run `python -m data.sources.schwab_auth` once "
                "to do the initial browser-based OAuth flow."
            )
        if self._is_access_token_expired():
            self._refresh_access_token()
        return self._token_data["access_token"]

    def run_initial_oauth(self) -> None:
        """Open browser, prompt user to paste redirect URL, exchange code for tokens."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.callback_url,
            "response_type": "code",
        }
        auth_url = f"{_AUTHORIZE_URL}?{urlencode(params)}"

        print("\n" + "=" * 70)
        print("Schwab OAuth — Step 1 / 2")
        print("=" * 70)
        print(f"\nOpening this URL in your browser:\n  {auth_url}\n")
        print("Log in with your Schwab brokerage credentials (NOT API).")
        print("After clicking 'Allow', your browser will be redirected to a URL")
        print("that probably won't load (that's expected).")
        print(
            f"\nCOPY THE FULL URL from your browser address bar after redirect,\n"
            f"then paste it below.\n"
        )

        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        redirected_url = input("Paste redirected URL here: ").strip()
        parsed = urlparse(redirected_url)
        qs = parse_qs(parsed.query)
        if "code" not in qs:
            raise SchwabAuthError(
                f"Pasted URL does not contain ?code=… ({redirected_url!r})"
            )
        code = qs["code"][0]

        self._exchange_code_for_tokens(code)
        print("\n  Token saved to:", self.token_path)
        print("  Access token expires in ~30 min, refreshed automatically.")
        print("  Refresh token expires in 7 days — re-run this script then.\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _basic_auth_header(self) -> str:
        creds = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(creds).decode("ascii")

    def _load_token_if_exists(self) -> None:
        if not self.token_path.exists():
            return
        try:
            self._token_data = json.loads(self.token_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load token file %s: %s", self.token_path, exc)
            self._token_data = None

    def _save_token(self) -> None:
        assert self._token_data is not None
        self.token_path.write_text(json.dumps(self._token_data, indent=2))
        # chmod 600 so it's user-only readable
        os.chmod(self.token_path, 0o600)

    def _is_access_token_expired(self) -> bool:
        assert self._token_data is not None
        # Treat token as expired 60s before its actual deadline for safety.
        exp_ts = self._token_data.get("access_token_expires_at", 0)
        return time.time() + 60 > exp_ts

    def _exchange_code_for_tokens(self, code: str) -> None:
        resp = requests.post(
            _TOKEN_URL,
            headers={
                "Authorization": self._basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.callback_url,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise SchwabAuthError(
                f"Schwab token exchange failed ({resp.status_code}): {resp.text[:300]}"
            )
        body = resp.json()
        now = time.time()
        self._token_data = {
            "access_token": body["access_token"],
            "refresh_token": body["refresh_token"],
            "access_token_expires_at": now + body.get("expires_in", 1800),
            # Schwab refresh tokens are valid 7 days; record issuance for visibility
            "refresh_token_issued_at": now,
        }
        self._save_token()

    def _refresh_access_token(self) -> None:
        assert self._token_data is not None
        resp = requests.post(
            _TOKEN_URL,
            headers={
                "Authorization": self._basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._token_data["refresh_token"],
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise SchwabAuthError(
                f"Schwab token refresh failed ({resp.status_code}): {resp.text[:300]}. "
                f"Refresh token may have expired (7-day limit) — re-run "
                f"`python -m data.sources.schwab_auth` to re-authorize."
            )
        body = resp.json()
        now = time.time()
        self._token_data["access_token"] = body["access_token"]
        self._token_data["access_token_expires_at"] = now + body.get("expires_in", 1800)
        # Schwab returns a new refresh_token on each refresh too
        if "refresh_token" in body:
            self._token_data["refresh_token"] = body["refresh_token"]
            self._token_data["refresh_token_issued_at"] = now
        self._save_token()


if __name__ == "__main__":
    # One-time: open browser to authorize, capture redirect, save tokens.
    try:
        from dotenv import load_dotenv

        load_dotenv("/Users/sarthakhans/options-scanner/.env")
    except ImportError:
        pass

    auth = SchwabAuth()
    auth.run_initial_oauth()

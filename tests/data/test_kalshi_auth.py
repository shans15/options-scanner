from __future__ import annotations

import base64
import time

import pytest

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from data.sources.kalshi_auth import KalshiAuthError, KalshiSigner


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

def test_signer_raises_when_access_key_missing(monkeypatch):
    monkeypatch.delenv("KALSHI_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "/tmp/anything.pem")
    with pytest.raises(KalshiAuthError, match="ACCESS_KEY_ID"):
        KalshiSigner()


def test_signer_raises_when_key_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KALSHI_ACCESS_KEY_ID", "test-uuid")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(tmp_path / "nonexistent.pem"))
    with pytest.raises(KalshiAuthError, match="not found"):
        KalshiSigner()


def test_signer_raises_when_placeholder_value(monkeypatch):
    monkeypatch.setenv("KALSHI_ACCESS_KEY_ID", "PASTE_YOUR_ACCESS_KEY_ID_HERE")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "/tmp/x.pem")
    with pytest.raises(KalshiAuthError):
        KalshiSigner()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def _make_key_file(tmp_path):
    """Generate an RSA private key and write it as PEM; return (key, path)."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_file = tmp_path / "test.pem"
    key_file.write_bytes(pem)
    return key, key_file


def test_sign_headers_produces_three_headers(tmp_path, monkeypatch):
    # Clear env so KalshiSigner uses explicit args only
    monkeypatch.delenv("KALSHI_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)

    key, key_file = _make_key_file(tmp_path)
    signer = KalshiSigner(access_key_id="test-uuid", private_key_path=str(key_file))
    headers = signer.sign_headers("GET", "/trade-api/v2/markets")

    assert set(headers.keys()) == {
        "KALSHI-ACCESS-KEY",
        "KALSHI-ACCESS-TIMESTAMP",
        "KALSHI-ACCESS-SIGNATURE",
    }
    assert headers["KALSHI-ACCESS-KEY"] == "test-uuid"
    # Timestamp must be parseable as int and within last 5 seconds
    ts = int(headers["KALSHI-ACCESS-TIMESTAMP"])
    assert abs(time.time() * 1000 - ts) < 5000
    # Signature must be valid base64 with non-zero length
    sig_bytes = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    assert len(sig_bytes) > 0


def test_sign_headers_signature_verifies_with_public_key(tmp_path, monkeypatch):
    """Generate a key, sign with it, verify with its public key.

    Confirms our signing implementation produces valid RSA-PSS signatures.
    """
    monkeypatch.delenv("KALSHI_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)

    key, key_file = _make_key_file(tmp_path)
    signer = KalshiSigner(access_key_id="test-uuid", private_key_path=str(key_file))
    headers = signer.sign_headers("GET", "/trade-api/v2/markets?status=open")

    # Reconstruct the message and verify
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    msg = f"{ts}GET/trade-api/v2/markets?status=open".encode("utf-8")
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])

    # If verify() does not raise, the signature is cryptographically valid
    key.public_key().verify(
        sig,
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

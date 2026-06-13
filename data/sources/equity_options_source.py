from __future__ import annotations

from data.sources.base import RawContract
from data.sources.yahooquery_source import YahooQuerySource
from data.sources.yfinance_source import YfinanceSource
from data.fallback import fetch_with_fallback


# ---------------------------------------------------------------------------
# Crypto-equity proxies with weekly options available to US-based traders.
# Each entry: (ticker, approx_correlation_to_crypto, description)
# Correlation values are approximate and should be verified before trading.
# ---------------------------------------------------------------------------
CRYPTO_PROXIES: dict[str, list[tuple[str, float, str]]] = {
    "BTC": [
        ("MSTR", 0.95, "MicroStrategy — large BTC treasury position; high beta"),
        ("IBIT", 0.99, "BlackRock iShares Bitcoin Trust ETF — most direct spot proxy"),
        ("BITO", 0.95, "ProShares Bitcoin Strategy ETF — futures-based, slight roll drag"),
        ("GBTC", 0.98, "Grayscale Bitcoin Trust ETF — direct spot proxy"),
        ("COIN", 0.80, "Coinbase — crypto exchange; broader crypto beta vs pure BTC"),
    ],
    "ETH": [
        ("ETHA", 0.99, "BlackRock iShares Ethereum Trust ETF — most direct spot proxy"),
        ("ETHE", 0.97, "Grayscale Ethereum Trust ETF — direct spot proxy"),
        ("COIN", 0.80, "Coinbase — crypto exchange; broader crypto beta vs pure ETH"),
    ],
}


def fetch_proxy_chains(crypto: str) -> dict[str, list[RawContract]]:
    """Fetch option chains for all configured proxies of a given crypto.

    Uses yahooquery as the primary source and yfinance as a fallback. Both
    sources implement the same fetch_option_chain() interface defined in
    data.sources.base.DataSource.

    Parameters
    ----------
    crypto : str
        'BTC' or 'ETH'. Must be a key in CRYPTO_PROXIES.

    Returns
    -------
    dict[str, list[RawContract]]
        Mapping of ticker → list of RawContract objects from the standard
        scanner option-chain sources. Tickers that fail entirely return an
        empty list rather than raising.

    Raises
    ------
    ValueError
        If crypto is not a recognised key in CRYPTO_PROXIES.
    """
    crypto = crypto.upper()
    if crypto not in CRYPTO_PROXIES:
        raise ValueError(
            f"Unknown crypto '{crypto}'. "
            f"Available: {sorted(CRYPTO_PROXIES.keys())}"
        )

    yq_source = YahooQuerySource()
    yf_source = YfinanceSource()
    sources = [yq_source, yf_source]

    results: dict[str, list[RawContract]] = {}
    for ticker, _corr, _desc in CRYPTO_PROXIES[crypto]:
        try:
            contracts: list[RawContract] = fetch_with_fallback(
                sources, "fetch_option_chain", ticker
            )
        except Exception:
            # Fail gracefully: a missing proxy should not abort the entire scan
            contracts = []
        results[ticker] = contracts

    return results


def get_proxy_spot(ticker: str) -> float:
    """Fetch current spot price for a proxy ticker.

    Tries yahooquery first, then yfinance.

    Returns
    -------
    float
        Last traded price, or 0.0 if all sources fail.
    """
    yq_source = YahooQuerySource()
    yf_source = YfinanceSource()
    try:
        return fetch_with_fallback([yq_source, yf_source], "fetch_spot", ticker)
    except Exception:
        return 0.0

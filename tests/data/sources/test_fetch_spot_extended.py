from unittest.mock import patch, MagicMock
from data.sources.yahooquery_source import YahooQuerySource
from data.sources.yfinance_source import YfinanceSource


def test_yahooquery_returns_premarket_during_pre_session():
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'preMarketPrice': 733.98, 'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY') == 733.98


def test_yahooquery_returns_regular_when_no_premarket_field():
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY') == 744.39  # Falls through


def test_yahooquery_during_regular_session_uses_regular_price():
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'preMarketPrice': 733.98, 'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='regular'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY') == 744.39


def test_prefer_extended_false_uses_regular_in_pre_market():
    """When prefer_extended=False, should use regular price even pre-market."""
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'preMarketPrice': 733.98, 'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY', prefer_extended=False) == 744.39


def test_yahooquery_returns_postmarket_during_post_session():
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'postMarketPrice': 741.50, 'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='post_market'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY') == 741.50


def test_yahooquery_post_market_falls_through_when_no_post_price():
    with patch('data.sources.yahooquery_source.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.price = {'SPY': {'regularMarketPrice': 744.39}}
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='post_market'):
            src = YahooQuerySource()
            assert src.fetch_spot('SPY') == 744.39


def test_schwab_requests_extended_fields():
    """Verify Schwab source requests the 'extended' field block."""
    from data.sources.schwab_source import SchwabSource

    with patch.object(SchwabSource, '__init__', return_value=None):
        src = SchwabSource()
        src._request = MagicMock(return_value={
            'SPY': {
                'quote': {'lastPrice': 744.39, 'bidPrice': 744.30, 'askPrice': 744.45},
                'extended': {'lastPrice': 733.98, 'bidPrice': 733.90, 'askPrice': 734.05}
            }
        })
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            spot = src.fetch_spot('SPY')
            # Should call _request with fields=quote,extended,regular
            src._request.assert_called_once()
            args, kwargs = src._request.call_args
            # Check params were passed
            params = kwargs.get('params') or (args[1] if len(args) > 1 else None)
            assert params is not None
            assert 'extended' in params.get('fields', '')
            # Should return extended mid: (733.90 + 734.05) / 2 = 733.975
            assert abs(spot - 733.975) < 0.01


def test_schwab_during_regular_uses_quote_block():
    from data.sources.schwab_source import SchwabSource

    with patch.object(SchwabSource, '__init__', return_value=None):
        src = SchwabSource()
        src._request = MagicMock(return_value={
            'SPY': {
                'quote': {'lastPrice': 744.39, 'bidPrice': 744.30, 'askPrice': 744.45},
                'extended': {'lastPrice': 733.98, 'bidPrice': 733.90, 'askPrice': 734.05}
            }
        })
        with patch('data.sources.market_session.current_session', return_value='regular'):
            spot = src.fetch_spot('SPY')
            # Should return regular mid: (744.30 + 744.45) / 2 = 744.375
            assert abs(spot - 744.375) < 0.01


def test_schwab_prefer_extended_false_uses_quote_during_pre_market():
    from data.sources.schwab_source import SchwabSource

    with patch.object(SchwabSource, '__init__', return_value=None):
        src = SchwabSource()
        src._request = MagicMock(return_value={
            'SPY': {
                'quote': {'lastPrice': 744.39, 'bidPrice': 744.30, 'askPrice': 744.45},
                'extended': {'lastPrice': 733.98, 'bidPrice': 733.90, 'askPrice': 734.05}
            }
        })
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            spot = src.fetch_spot('SPY', prefer_extended=False)
            # prefer_extended=False → use regular quote block
            assert abs(spot - 744.375) < 0.01


def test_schwab_extended_falls_back_to_last_price_when_no_bid_ask():
    """When extended block has no bid/ask but has lastPrice, use lastPrice."""
    from data.sources.schwab_source import SchwabSource

    with patch.object(SchwabSource, '__init__', return_value=None):
        src = SchwabSource()
        src._request = MagicMock(return_value={
            'SPY': {
                'quote': {'lastPrice': 744.39, 'bidPrice': 744.30, 'askPrice': 744.45},
                'extended': {'lastPrice': 733.98, 'bidPrice': 0, 'askPrice': 0}
            }
        })
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            spot = src.fetch_spot('SPY')
            assert abs(spot - 733.98) < 0.01


def test_schwab_extended_empty_falls_back_to_regular_quote():
    """When extended block is empty, fall through to regular quote."""
    from data.sources.schwab_source import SchwabSource

    with patch.object(SchwabSource, '__init__', return_value=None):
        src = SchwabSource()
        src._request = MagicMock(return_value={
            'SPY': {
                'quote': {'lastPrice': 744.39, 'bidPrice': 744.30, 'askPrice': 744.45},
                'extended': {}
            }
        })
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            spot = src.fetch_spot('SPY')
            assert abs(spot - 744.375) < 0.01


def test_yfinance_returns_premarket_during_pre_session():
    with patch('data.sources.yfinance_source.yf.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.info = {'preMarketPrice': 733.98}
        mock_t.fast_info = MagicMock(last_price=744.39)
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YfinanceSource()
            assert src.fetch_spot('SPY') == 733.98


def test_yfinance_returns_postmarket_during_post_session():
    with patch('data.sources.yfinance_source.yf.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.info = {'postMarketPrice': 741.50}
        mock_t.fast_info = MagicMock(last_price=744.39)
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='post_market'):
            src = YfinanceSource()
            assert src.fetch_spot('SPY') == 741.50


def test_yfinance_falls_back_to_fast_info_when_info_raises():
    with patch('data.sources.yfinance_source.yf.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        type(mock_t).info = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
        mock_t.fast_info = MagicMock(last_price=744.39)
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YfinanceSource()
            assert src.fetch_spot('SPY') == 744.39


def test_yfinance_prefer_extended_false_uses_fast_info():
    with patch('data.sources.yfinance_source.yf.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.info = {'preMarketPrice': 733.98}
        mock_t.fast_info = MagicMock(last_price=744.39)
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='pre_market'):
            src = YfinanceSource()
            assert src.fetch_spot('SPY', prefer_extended=False) == 744.39


def test_yfinance_regular_session_uses_fast_info():
    with patch('data.sources.yfinance_source.yf.Ticker') as mock_ticker_cls:
        mock_t = MagicMock()
        mock_t.info = {'preMarketPrice': 733.98}
        mock_t.fast_info = MagicMock(last_price=744.39)
        mock_ticker_cls.return_value = mock_t
        with patch('data.sources.market_session.current_session', return_value='regular'):
            src = YfinanceSource()
            assert src.fetch_spot('SPY') == 744.39

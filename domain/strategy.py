from __future__ import annotations
from abc import ABC, abstractmethod
from typing import ClassVar, Literal
from dataclasses import dataclass

from domain.contract import Contract


@dataclass
class Strategy(ABC):
    name: ClassVar[str]
    direction: ClassVar[Literal['sell', 'buy']]
    option_type: ClassVar[Literal['put', 'call']]
    delta_min: ClassVar[float]
    delta_max: ClassVar[float]

    def applies_to(self, c: Contract) -> bool:
        if c.option_type != self.option_type:
            return False
        return self.delta_min <= abs(c.delta) <= self.delta_max

    @abstractmethod
    def breakeven(self, c: Contract) -> float: ...

    @abstractmethod
    def profit_condition(self, S_T: float, c: Contract) -> bool: ...

    @abstractmethod
    def expected_value(self, c: Contract, pop_blended: float, max_adverse_loss: float, expected_profit_when_itm: float = 0.0) -> float: ...

    @abstractmethod
    def margin_estimate(self, c: Contract) -> float: ...


class NakedPut(Strategy):
    name = 'naked_put'
    direction = 'sell'
    option_type = 'put'
    delta_min = 0.16
    delta_max = 0.30

    def breakeven(self, c: Contract) -> float:
        return c.strike - c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T > c.strike

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return c.mid * pop_blended - max_adverse_loss * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        otm = abs(c.spot_price - c.strike)
        return round(max(0.20 * c.spot_price - otm + c.mid, 0.10 * c.strike + c.mid) * 100, 2)


class NakedCall(Strategy):
    name = 'naked_call'
    direction = 'sell'
    option_type = 'call'
    delta_min = 0.16
    delta_max = 0.30

    def breakeven(self, c: Contract) -> float:
        return c.strike + c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T < c.strike

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return c.mid * pop_blended - max_adverse_loss * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        otm = abs(c.spot_price - c.strike)
        return round(max(0.20 * c.spot_price - otm + c.mid, 0.10 * c.spot_price + c.mid) * 100, 2)


class LongPut(Strategy):
    name = 'long_put'
    direction = 'buy'
    option_type = 'put'
    delta_min = 0.40
    delta_max = 0.60

    def breakeven(self, c: Contract) -> float:
        return c.strike - c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T < c.strike - c.mid

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return expected_profit_when_itm * pop_blended - c.mid * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        return round(c.mid * 100, 2)


class LongCall(Strategy):
    name = 'long_call'
    direction = 'buy'
    option_type = 'call'
    delta_min = 0.40
    delta_max = 0.60

    def breakeven(self, c: Contract) -> float:
        return c.strike + c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T > c.strike + c.mid

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return expected_profit_when_itm * pop_blended - c.mid * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        return round(c.mid * 100, 2)


ALL_STRATEGIES: list[Strategy] = [NakedPut(), NakedCall(), LongPut(), LongCall()]

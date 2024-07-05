import numpy as np
import pandas as pd


from syscore.exceptions import missingData

from sysdata.config.configdata import Config
from sysdata.sim.sim_data import simData

from systems.stage import SystemStage
from systems.system_cache import input, diagnostic, output
from private.systems.orion.rawdata.rawdata import OrionRawData as RawData


class OrionPositionSizing(SystemStage):
    @property
    def name(self):
        return "positionSize"

    @output()
    def get_subsystem_position(self, instrument_code: str) -> pd.Series:
        """
        Get scaled position (assuming for now we trade our entire capital for one instrument)

        KEY OUTPUT

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :returns: Tx1 pd.DataFrame

        >>> from systems.tests.testdata import get_test_object_futures_with_comb_forecasts
        >>> from systems.basesystem import System
        >>> (comb, fcs, rules, rawdata, data, config)=get_test_object_futures_with_comb_forecasts()
        >>> system=System([rawdata, rules, fcs, comb, PositionSizing()], data, config)
        >>>
        >>> system.positionSize.get_subsystem_position("EDOLLAR").tail(2)
                    ss_position
        2015-12-10     1.811465
        2015-12-11     2.544598
        >>>
        >>> system2=System([rawdata, rules, fcs, comb, PositionSizing()], data, config)
        >>> system2.positionSize.get_subsystem_position("EDOLLAR").tail(2)
                    ss_position
        2015-12-10     1.811465
        2015-12-11     2.544598

        """
        self.log.debug(
            "Calculating subsystem position for %s" % instrument_code,
            instrument_code=instrument_code,
        )
        """
        We don't allow this to be changed in config
        """

        forecast = self.get_forecast(instrument_code)
        risk_per_trade_pct_capital = self.get_risk_per_trade_pct_capital()
        capital_allocated_to_instrument = self.get_capital_allocated_to_instrument(instrument_code)

        risk_per_trade_currency = risk_per_trade_pct_capital * capital_allocated_to_instrument

        stop_loss_level = self.get_stop_loss_levels(instrument_code)
        price = self.get_underlying_price(instrument_code)
        multiplier = self.rawdata_stage.get_value_of_block_price_move(instrument_code)

        risk_currency = np.sign(forecast) * (price['FINAL'] - stop_loss_level) * multiplier
        subsystem_position_raw = risk_per_trade_currency / risk_currency
        subsystem_position = self._apply_long_only_constraint_to_position(
            position=subsystem_position_raw, instrument_code=instrument_code
        )
        subsystem_position = subsystem_position.round(0)

        return subsystem_position

    def get_capital_allocated_to_instrument(self, instrument_code: str):
        notional_trading_capital = self.get_notional_trading_capital()
        instruments_weight = self.config.instrument_weights[instrument_code]
        capital_allocated_to_instrument = notional_trading_capital * instruments_weight

        return capital_allocated_to_instrument

    def get_risk_per_trade_pct_capital(self):
        return self.config.get_element_or_default("risk_per_trade_pct_capital", 0)

    def get_stop_loss_levels(self, instrument_code: str):
        strategy_outputs = self.get_strategy_outputs(instrument_code)
        return strategy_outputs['stop_loss_levels_after_slpt']

    def get_forecast(self, instrument_code: str):
        strategy_outputs = self.get_strategy_outputs(instrument_code)
        return strategy_outputs['forecasts']

    def get_strategy_outputs(self, instrument_code: str):
        pathdependency_stage = self.pathdependency_stage()
        return pathdependency_stage.get_signals_after_limit_price_is_hit_stop_loss_and_profit_target(
            instrument_code
        )

    def pathdependency_stage(self):
        try:
            pathdependency_stage = getattr(self.parent, "pathdependency")
        except AttributeError as e:
            raise missingData from e

        return pathdependency_stage

    def _apply_long_only_constraint_to_position(
        self, position: pd.Series, instrument_code: str
    ) -> pd.Series:
        instrument_long_only = self._is_instrument_long_only(instrument_code)
        if instrument_long_only:
            position[position < 0.0] = 0.0

        return position

    @diagnostic()
    def _is_instrument_long_only(self, instrument_code: str) -> bool:
        list_of_long_only_instruments = self._get_list_of_long_only_instruments()

        return instrument_code in list_of_long_only_instruments

    @diagnostic()
    def _get_list_of_long_only_instruments(self) -> list:
        config = self.config
        long_only = config.get_element_or_default("long_only_instruments", [])
        return long_only

    @property
    def config(self) -> Config:
        return self.parent.config

    @diagnostic()
    def get_block_value(self, instrument_code: str) -> pd.Series:
        """
        Calculate block value for instrument_code

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :returns: Tx1 pd.DataFrame

        >>> from systems.tests.testdata import get_test_object_futures_with_comb_forecasts
        >>> from systems.basesystem import System
        >>> (comb, fcs, rules, rawdata, data, config)=get_test_object_futures_with_comb_forecasts()
        >>> system=System([rawdata, rules, fcs, comb, PositionSizing()], data, config)
        >>>
        >>> system.positionSize.get_block_value("EDOLLAR").tail(2)
                       bvalue
        2015-12-10  2447.0000
        2015-12-11  2449.6875
        >>>
        >>> system=System([rules, fcs, comb, PositionSizing()], data, config)
        >>> system.positionSize.get_block_value("EDOLLAR").tail(2)
                       bvalue
        2015-12-10  2447.0000
        2015-12-11  2449.6875

        """

        underlying_price = self.get_underlying_price(instrument_code)
        value_of_price_move = self.rawdata_stage.get_value_of_block_price_move(
            instrument_code
        )

        block_value = underlying_price['FINAL'].ffill() * value_of_price_move * 0.01

        return block_value

    @diagnostic()
    def get_underlying_price(self, instrument_code: str) -> pd.Series:
        """
        Get various things from data and rawdata to calculate position sizes

        KEY INPUT

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :returns: Tx1 pd.DataFrame: underlying price [as used to work out % volatility],

        >>> from systems.tests.testdata import get_test_object_futures_with_comb_forecasts
        >>> from systems.basesystem import System
        >>> (comb, fcs, rules, rawdata, data, config)=get_test_object_futures_with_comb_forecasts()
        >>> system=System([rawdata, rules, fcs, comb, PositionSizing()], data, config)
        >>>
        >>> ans=system.positionSize.get_underlying_price("EDOLLAR")
        >>> ans[0].tail(2)
                      price
        2015-12-10  97.8800
        2015-12-11  97.9875
        >>>
        >>> ans[1]
        2500
        >>>
        >>> system=System([rules, fcs, comb, PositionSizing()], data, config)
        >>>
        >>> ans=system.positionSize.get_underlying_price("EDOLLAR")
        >>> ans[0].tail(2)
                      price
        2015-12-10  97.8800
        2015-12-11  97.9875
        >>>
        >>> ans[1]
        2500


        """
        try:
            rawdata = self.rawdata_stage
        except missingData:
            underlying_price = self.data.minute_prices(instrument_code)
        else:
            underlying_price = rawdata.daily_denominator_price(instrument_code)

        return underlying_price

    @property
    def rawdata_stage(self) -> RawData:
        try:
            rawdata_stage = getattr(self.parent, "rawdata")
        except AttributeError as e:
            raise missingData from e

        return rawdata_stage

    @property
    def data(self) -> simData:
        return self.parent.data

    @diagnostic()
    def get_notional_trading_capital(self) -> float:
        notional_trading_capital = float(self.config.notional_trading_capital)
        return notional_trading_capital

    @diagnostic()
    def get_base_currency(self) -> str:
        base_currency = self.config.base_currency
        return base_currency

    @input
    def get_fx_rate(self, instrument_code: str) -> pd.Series:
        """
        Get FX rate to translate instrument volatility into same currency as account value.

        KEY INPUT

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :returns: Tx1 pd.DataFrame: fx rate

        >>> from systems.tests.testdata import get_test_object_futures_with_comb_forecasts
        >>> from systems.basesystem import System
        >>> (comb, fcs, rules, rawdata, data, config)=get_test_object_futures_with_comb_forecasts()
        >>> system=System([rawdata, rules, fcs, comb, PositionSizing()], data, config)
        >>>
        >>> system.positionSize.get_fx_rate("EDOLLAR").tail(2)
                          fx
        2015-12-09  0.664311
        2015-12-10  0.660759

        """

        base_currency = self.get_base_currency()
        fx_rate = self.rawdata_stage.get_fx_for_instrument(
            instrument_code, base_currency
        )

        return fx_rate


if __name__ == "__main__":
    import doctest

    doctest.testmod()

from syscore.dateutils import BUSINESS_DAY_FREQ, HOURLY_FREQ, MINUTE_FREQ
from syscore.pandas.frequency import infer_frequency
from syscore.constants import arg_not_supplied
from sysobjects.instruments import instrumentCosts

from syscore.pandas.pdutils import from_scalar_values_to_ts

import pandas as pd
from systems.stage import SystemStage
from systems.system_cache import diagnostic


class accountInputs(SystemStage):
    def get_raw_price(self, instrument_code: str) -> pd.Series:
        return self.parent.data.get_raw_price(instrument_code)

    def get_instrument_prices_for_position_or_forecast(
        self, instrument_code: str, position_or_forecast: pd.Series = arg_not_supplied
    ) -> pd.Series:
        if position_or_forecast is arg_not_supplied:
            return self.get_daily_prices(instrument_code)

        instrument_prices = (
            self.instrument_prices_for_position_or_forecast_infer_frequency(
                instrument_code=instrument_code,
                position_or_forecast=position_or_forecast,
            )
        )
        instrument_prices_reindexed = instrument_prices.reindex(
            position_or_forecast.index, method="ffill"
        )

        return instrument_prices_reindexed

    def instrument_prices_for_position_or_forecast_infer_frequency(
        self, instrument_code: str, position_or_forecast: pd.Series = arg_not_supplied
    ) -> pd.Series:
        try:
            frequency = infer_frequency(position_or_forecast)
            if frequency is BUSINESS_DAY_FREQ:
                instrument_prices = self.get_daily_prices(instrument_code)
            elif frequency is HOURLY_FREQ:
                instrument_prices = self.get_hourly_prices(instrument_code)
            elif frequency is MINUTE_FREQ:
                instrument_prices = self.get_minute_prices(instrument_code)
            else:
                raise Exception(
                    "Frequency %s does not have prices for %s should be minute or hourly or daily"
                    % (str(frequency), instrument_code)
                )
        except:
            self.log.warning(
                "Going to index minute prices for %s to position_or_forecast might result in phantoms"
                % instrument_code
            )
            minute_prices = self.get_minute_prices(instrument_code)

            instrument_prices = minute_prices.reindex(position_or_forecast.index)

        return instrument_prices

    def get_minute_prices(self, instrument_code: str) -> pd.Series:
        return self.parent.rawdata.get_minute_prices(instrument_code)

    def has_same_rules_as_code(self, instrument_code):
        """
        Return instruments with same trading rules as this instrument

        KEY INPUT

        :param instrument_code:
        :type str:

        :returns: list of str

        """
        return self.parent.combForecast.has_same_rules_as_code(instrument_code)

    def get_raw_cost_data(self, instrument_code: str) -> instrumentCosts:
        return self.parent.rawdata.get_raw_cost_data(instrument_code)

    def get_value_of_block_price_move(self, instrument_code: str) -> float:
        return self.parent.rawdata.get_value_of_block_price_move(instrument_code)

    def get_fx_rate(self, instrument_code: str) -> pd.Series:
        return self.parent.positionSize.get_fx_rate(instrument_code)

    def get_subsystem_position(self, instrument_code: str) -> pd.Series:
        return self.parent.positionSize.get_subsystem_position(instrument_code)

    def get_notional_capital(self) -> float:
        """
        Get notional capital from the previous module

        KEY INPUT

        :returns: float

        >>> from systems.basesystem import System
        >>> from systems.tests.testdata import get_test_object_futures_with_portfolios
        >>> (portfolio, posobject, combobject, capobject, rules, rawdata, data, config)=get_test_object_futures_with_portfolios()
        >>> system=System([portfolio, posobject, combobject, capobject, rules, rawdata, Account()], data, config)
        >>>
        >>> system.accounts.get_notional_capital()
        100000.0
        """
        return self.parent.config.notional_trading_capital

    def get_notional_position(self, instrument_code: str) -> pd.Series:
        """
        Get the notional position from a previous module

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :returns: Tx1 pd.DataFrame

        KEY INPUT

        >>> from systems.basesystem import System
        >>> from systems.tests.testdata import get_test_object_futures_with_portfolios
        >>> (portfolio, posobject, combobject, capobject, rules, rawdata, data, config)=get_test_object_futures_with_portfolios()
        >>> system=System([portfolio, posobject, combobject, capobject, rules, rawdata, Account()], data, config)
        >>>
        """
        return self.parent.portfolio.get_notional_position(instrument_code)

    def get_instrument_weights(self):
        """
        Get instrument weights

        KEY INPUT

        :returns: Tx1 pd.DataFrame


        """

        return self.parent.portfolio.get_instrument_weights()

    def get_instrument_list(self) -> list:
        return self.parent.get_instrument_list()

    @property
    def config(self):
        return self.parent.config

    def specific_instrument_weight(self, instrument_code: str) -> pd.Series:
        instrument_weights = self.instrument_weights()

        return instrument_weights[instrument_code]

    def instrument_weights(self) -> pd.DataFrame:
        return self.parent.portfolio.get_instrument_weights()

    def list_of_rules_for_code(self, instrument_code: str) -> list:
        return self.parent.combForecast.get_trading_rule_list(instrument_code)

    def list_of_trading_rules(self) -> list:
        return self.parent.rules.trading_rules()

    def get_actual_position(self, instrument_code: str) -> pd.Series:
        return self.parent.portfolio.get_actual_position(instrument_code)

    def get_rolls_per_year(self, instrument_code: str) -> int:
        rolls_per_year = self.parent.rawdata.rolls_per_year(instrument_code)

        return rolls_per_year


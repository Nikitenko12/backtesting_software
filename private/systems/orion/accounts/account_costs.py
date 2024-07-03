import pandas as pd

from syscore.maths import calculate_weighted_average_with_nans
from syscore.genutils import str2Bool
from syscore.dateutils import ROOT_BDAYS_INYEAR
from syscore.pandas.strategy_functions import turnover

from sysquant.estimators.turnover import turnoverDataForTradingRule

from systems.system_cache import diagnostic, input
from private.systems.orion.accounts.account_inputs import accountInputs


class accountCosts(accountInputs):
    @diagnostic()
    def _date_one_year_before_end_of_price_index(self, instrument_code: str):
        daily_price = self.get_instrument_prices_for_position_or_forecast(
            instrument_code
        )

        last_date = daily_price.index[-1]
        start_date = last_date - pd.DateOffset(years=1)

        return start_date

    @property
    def use_SR_costs(self) -> bool:
        return str2Bool(self.config.use_SR_costs)

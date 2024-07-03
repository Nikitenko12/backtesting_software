import pandas as pd

from syscore.pandas.strategy_functions import turnover

from systems.system_cache import diagnostic, dont_cache
from private.systems.orion.accounts.account_costs import accountCosts
from private.systems.orion.accounts.pandl_calculators.pandl_SR_cost import pandlCalculationWithSRCosts
from private.systems.orion.accounts.pandl_calculators.pandl_cash_costs import (
    pandlCalculationWithCashCostsAndFills,
)
from private.systems.orion.accounts.curves.account_curve import accountCurve


class accountInstruments(accountCosts):
    # dont' cache: just a switch method
    @dont_cache
    def pandl_for_instrument(
        self, instrument_code: str, delayfill: bool = True, roundpositions: bool = True
    ) -> accountCurve:
        """
        Get the p&l for one instrument

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :param delayfill: Lag fills by one day
        :type delayfill: bool

        :param roundpositions: Round positions to whole contracts
        :type roundpositions: bool

        :returns: accountCurve

        >>> from systems.basesystem import System
        >>> from systems.tests.testdata import get_test_object_futures_with_portfolios
        >>> (portfolio, posobject, combobject, capobject, rules, rawdata, data, config)=get_test_object_futures_with_portfolios()
        >>> system=System([portfolio, posobject, combobject, capobject, rules, rawdata, Account()], data, config)
        >>> system.accounts.pandl_for_instrument("US10").ann_std()
        0.13908407620762306
        """

        self.log.debug(
            "Calculating pandl for instrument for %s" % instrument_code,
            instrument_code=instrument_code,
        )

        positions = self.get_actual_position(
            instrument_code
        )

        instrument_pandl = self._pandl_for_instrument_with_positions(
            instrument_code,
            positions=positions,
            delayfill=delayfill,
            roundpositions=roundpositions,
        )

        return instrument_pandl

    @dont_cache
    def _pandl_for_instrument_with_positions(
        self,
        instrument_code: str,
        positions: pd.Series,
        delayfill: bool = True,
        roundpositions: bool = True,
    ) -> accountCurve:
        """
        Get the p&l for one instrument

        :param instrument_code: instrument to get values for
        :type instrument_code: str

        :param delayfill: Lag fills by one day
        :type delayfill: bool

        :param roundpositions: Round positions to whole contracts
        :type roundpositions: bool

        :returns: accountCurve

        >>> from systems.basesystem import System
        >>> from systems.tests.testdata import get_test_object_futures_with_portfolios
        >>> (portfolio, posobject, combobject, capobject, rules, rawdata, data, config)=get_test_object_futures_with_portfolios()
        >>> system=System([portfolio, posobject, combobject, capobject, rules, rawdata, Account()], data, config)
        >>> system.accounts.pandl_for_instrument("US10").ann_std()
        0.13908407620762306
        """

        self.log.debug(
            "Calculating pandl for instrument for %s" % instrument_code,
            instrument_code=instrument_code,
        )

        instrument_pandl = self._pandl_for_instrument_with_cash_costs(
            instrument_code,
            positions=positions,
            roundpositions=roundpositions,
            delayfill=delayfill,
        )

        return instrument_pandl

    @diagnostic(not_pickable=True)
    def _pandl_for_instrument_with_cash_costs(
        self,
        instrument_code: str,
        positions: pd.Series,
        delayfill: bool = True,
        roundpositions: bool = True,
    ) -> accountCurve:
        if not roundpositions:
            self.log.warning(
                "Using roundpositions=False with cash costs may lead to inaccurate costs (fixed costs, eg commissions will be overstated!!!"
            )

        raw_costs = self.get_raw_cost_data(instrument_code)

        price = self.get_instrument_prices_for_position_or_forecast(
            instrument_code, position_or_forecast=positions
        )
        fx = self.get_fx_rate(instrument_code)
        value_of_price_point = self.get_value_of_block_price_move(instrument_code)

        capital = self.get_notional_capital()

        vol_normalise_currency_costs = self.config.vol_normalise_currency_costs
        rolls_per_year = self.get_rolls_per_year(instrument_code)
        multiply_roll_costs_by = self.config.multiply_roll_costs_by

        pandl_calculator = pandlCalculationWithCashCostsAndFills(
            price,
            raw_costs=raw_costs,
            positions=positions,
            capital=capital,
            value_per_point=value_of_price_point,
            delayfill=delayfill,
            fx=fx,
            roundpositions=roundpositions,
            vol_normalise_currency_costs=vol_normalise_currency_costs,
            rolls_per_year=rolls_per_year,
            multiply_roll_costs_by=multiply_roll_costs_by,
        )

        account_curve = accountCurve(pandl_calculator, weighted=True)

        return account_curve

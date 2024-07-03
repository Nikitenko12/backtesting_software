import pandas as pd

from syscore.dateutils import ROOT_BDAYS_INYEAR
from syscore.constants import arg_not_supplied
from syscore.pandas.pdutils import sum_series
from sysquant.estimators.vol import robust_daily_vol_given_price

from systems.system_cache import diagnostic
from private.systems.orion.accounts.account_costs import accountCosts
from private.systems.orion.accounts.pandl_calculators.pandl_SR_cost import pandlCalculationWithSRCosts
from private.systems.orion.accounts.curves.account_curve import accountCurve


class accountForecast(accountCosts):
    pass


ARBITRARY_FORECAST_CAPITAL = 100
ARBITRARY_FORECAST_ANNUAL_RISK_TARGET_PERCENTAGE = 0.16


ARBITRARY_VALUE_OF_PRICE_POINT = 1.0


def pandl_for_instrument_forecast(
    forecast: pd.Series,
    price: pd.Series,
    capital: float = ARBITRARY_FORECAST_CAPITAL,
    fx=arg_not_supplied,
    risk_target: float = ARBITRARY_FORECAST_ANNUAL_RISK_TARGET_PERCENTAGE,
    daily_returns_volatility: pd.Series = arg_not_supplied,
    target_abs_forecast: float = 10.0,
    SR_cost=0.0,
    delayfill=True,
    value_per_point=ARBITRARY_VALUE_OF_PRICE_POINT,
) -> accountCurve:
    if daily_returns_volatility is arg_not_supplied:
        daily_returns_volatility = robust_daily_vol_given_price(price)

    normalised_forecast = _get_normalised_forecast(
        forecast, target_abs_forecast=target_abs_forecast
    )

    average_notional_position = _get_average_notional_position(
        daily_returns_volatility,
        risk_target=risk_target,
        value_per_point=value_per_point,
        capital=capital,
    )

    notional_position = _get_notional_position_for_forecast(
        normalised_forecast, average_notional_position=average_notional_position
    )

    pandl_calculator = pandlCalculationWithSRCosts(
        price,
        SR_cost=SR_cost,
        positions=notional_position,
        fx=fx,
        daily_returns_volatility=daily_returns_volatility,
        average_position=average_notional_position,
        capital=capital,
        value_per_point=value_per_point,
        delayfill=delayfill,
    )

    account_curve = accountCurve(pandl_calculator)

    return account_curve


def _get_notional_position_for_forecast(
    normalised_forecast: pd.Series, average_notional_position: pd.Series
) -> pd.Series:
    aligned_average = average_notional_position.reindex(
        normalised_forecast.index, method="ffill"
    )

    return aligned_average * normalised_forecast


def _get_average_notional_position(
    daily_returns_volatility: pd.Series,
    capital: float = ARBITRARY_FORECAST_CAPITAL,
    risk_target: float = ARBITRARY_FORECAST_ANNUAL_RISK_TARGET_PERCENTAGE,
    value_per_point=ARBITRARY_VALUE_OF_PRICE_POINT,
) -> pd.Series:
    daily_risk_target = risk_target / ROOT_BDAYS_INYEAR
    daily_cash_vol_target = capital * daily_risk_target

    instrument_currency_vol = daily_returns_volatility * value_per_point
    average_notional_position = daily_cash_vol_target / instrument_currency_vol

    return average_notional_position


def _get_normalised_forecast(
    forecast: pd.Series, target_abs_forecast: float = 10.0
) -> pd.Series:
    normalised_forecast = forecast / target_abs_forecast

    return normalised_forecast

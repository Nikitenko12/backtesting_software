import pandas as pd

from private.systems.orion.accounts.order_simulator.account_curve_order_simulator import (
    AccountWithOrderSimulator,
)
from private.systems.orion.accounts.order_simulator.pandl_order_simulator import (
    OrderSimulator,
    OrdersSeriesData,
)
from systems.system_cache import diagnostic


class MinuteOrderSimulatorOfMarketOrders(OrderSimulator):
    def _series_data(self) -> OrdersSeriesData:
        series_data = _build_minute_series_data_for_order_simulator(
            system_accounts_stage=self.system_accounts_stage,  # ignore type hint
            instrument_code=self.instrument_code,
            is_subsystem=self.is_subsystem,
        )
        return series_data


def _build_minute_series_data_for_order_simulator(
    system_accounts_stage,  ## no explicit type would cause circular import
    instrument_code: str,
    is_subsystem: bool = False,
) -> OrdersSeriesData:
    price_series = system_accounts_stage.get_minute_prices(instrument_code)
    if is_subsystem:
        unrounded_positions = (
            system_accounts_stage.get_unrounded_subsystem_position_for_order_simulator(
                instrument_code
            )
        )
    else:
        unrounded_positions = (
            system_accounts_stage.get_unrounded_instrument_position_for_order_simulator(
                instrument_code
            )
        )
    path_dependency_outputs = system_accounts_stage.parent.positionSize.get_strategy_outputs(instrument_code)
    long_limit_price_series = path_dependency_outputs['long_limit_prices_after_slpt']
    short_limit_price_series = path_dependency_outputs['short_limit_prices_after_slpt']

    price_series = price_series.sort_index()
    unrounded_positions = unrounded_positions.sort_index()

    all_index = pd.concat([price_series, unrounded_positions, long_limit_price_series, short_limit_price_series], axis=1).index

    price_series = price_series.reindex(all_index).ffill()
    unrounded_positions = unrounded_positions.reindex(all_index).ffill()
    long_limit_price_series = long_limit_price_series.reindex(all_index).ffill()
    short_limit_price_series = short_limit_price_series.reindex(all_index).ffill()

    series_data = OrdersSeriesData(
        price_series=price_series, unrounded_positions=unrounded_positions,
        long_limit_price_series=long_limit_price_series, short_limit_price_series=short_limit_price_series
    )
    return series_data


class AccountWithOrderSimulatorForMinuteMarketOrders(AccountWithOrderSimulator):
    @diagnostic(not_pickable=True)
    def get_order_simulator(
        self, instrument_code, is_subsystem: bool
    ) -> MinuteOrderSimulatorOfMarketOrders:
        order_simulator = MinuteOrderSimulatorOfMarketOrders(
            system_accounts_stage=self,
            instrument_code=instrument_code,
            is_subsystem=is_subsystem,
        )
        return order_simulator

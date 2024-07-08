import datetime
from typing import Tuple, Callable

from collections import namedtuple
import numpy as np


from sysobjects.fills import Fill, empty_fill
from private.systems.orion.accounts.order_simulator.fills_and_orders import (
    fill_list_of_simple_orders,
    empty_list_of_orders_with_no_fills,
    ListOfSimpleOrdersAndResultingFill,
)
from private.systems.orion.accounts.order_simulator.simple_orders import (
    ListOfSimpleOrdersWithDate,
    SimpleOrderWithDate,
)

from private.systems.orion.accounts.order_simulator.account_curve_order_simulator import (
    AccountWithOrderSimulator,
)
from private.systems.orion.accounts.order_simulator.pandl_order_simulator import (
    OrderSimulator, OrdersSeriesData
)
from private.systems.orion.accounts.order_simulator.minute_market_orders import (
    MinuteOrderSimulatorOfMarketOrders,
)
from systems.system_cache import diagnostic


class MinuteOrderSimulatorOfLimitOrders(MinuteOrderSimulatorOfMarketOrders):
    @property
    def orders_fills_function(self) -> Callable:
        return generate_order_and_fill_at_idx_point_for_limit_orders

    @property
    def idx_data_function(self) -> Callable:
        return get_order_sim_minute_data_at_idx_point


DataAtIDXPoint = namedtuple(
    "DataAtIDXPoint",
    [
        "current_optimal_position", "current_price", "next_price", "next_datetime", "long_limit_price", "short_limit_price",
        "profit_target_level", "stop_loss_level",
    ],
)


def generate_order_and_fill_at_idx_point_for_limit_orders(
    current_position: int,
    current_datetime: datetime.datetime,
    data_for_idx: DataAtIDXPoint,
) -> Tuple[ListOfSimpleOrdersWithDate, Fill]:
    current_optimal_position = data_for_idx.current_optimal_position
    if np.isnan(current_optimal_position):
        quantity = 0
    else:
        quantity = round(current_optimal_position) - current_position

    if quantity == 0:
        notional_datetime_for_empty_fill = data_for_idx.next_datetime
        return empty_list_of_orders_with_no_fills(
            fill_datetime=notional_datetime_for_empty_fill
        )

    fill_datetime = data_for_idx.next_datetime
    if current_optimal_position > 0:
        limit_price = data_for_idx.long_limit_price
    elif current_optimal_position < 0:
        limit_price = data_for_idx.short_limit_price
    else:
        limit_price = None

    current_price = data_for_idx.current_price
    market_price = current_price
    next_price = data_for_idx.next_price
    if current_position > 0 > quantity:   # Exiting long position
        if current_price <= data_for_idx.stop_loss_level:
            limit_price = current_price
            fill_datetime = current_datetime
        elif next_price >= data_for_idx.profit_target_level:
            limit_price = data_for_idx.profit_target_level
            market_price = next_price

    elif current_position < 0 < quantity: # Exiting short position
        if current_price >= data_for_idx.stop_loss_level:
            limit_price = current_price
            fill_datetime = current_datetime
        elif next_price <= data_for_idx.profit_target_level:
            limit_price = data_for_idx.profit_target_level
            market_price = next_price

    simple_order = SimpleOrderWithDate(
        quantity=quantity,
        submit_date=current_datetime,
        limit_price=limit_price,
    )
    list_of_orders = ListOfSimpleOrdersWithDate([simple_order])
    fill = fill_list_of_simple_orders(
        list_of_orders=list_of_orders,
        market_price=market_price,
        fill_datetime=fill_datetime,
    )

    return ListOfSimpleOrdersAndResultingFill(list_of_orders=list_of_orders, fill=fill)


def get_order_sim_minute_data_at_idx_point(
    idx: int, series_data: OrdersSeriesData
) -> DataAtIDXPoint:
    unrounded_positions = series_data.unrounded_positions
    prices = series_data.price_series
    long_limit_prices = series_data.long_limit_price_series
    short_limit_prices = series_data.short_limit_price_series
    profit_target_levels = series_data.profit_target_level_series
    stop_loss_levels = series_data.stop_loss_level_series

    current_optimal_position = unrounded_positions.iloc[idx]
    next_price = prices.iloc[idx + 1]
    current_price = prices.iloc[idx]
    next_datetime = unrounded_positions.index[idx + 1]
    long_limit_price = long_limit_prices.iloc[idx]
    short_limit_price = short_limit_prices.iloc[idx]
    profit_target_level = profit_target_levels.iloc[idx]
    stop_loss_level = stop_loss_levels.iloc[idx]

    return DataAtIDXPoint(
        current_optimal_position=current_optimal_position,
        next_datetime=next_datetime,
        next_price=next_price,
        current_price=current_price,
        long_limit_price=long_limit_price,
        short_limit_price=short_limit_price,
        profit_target_level=profit_target_level,
        stop_loss_level=stop_loss_level,
    )


class AccountWithOrderSimulatorForLimitOrders(AccountWithOrderSimulator):
    @diagnostic(not_pickable=True)
    def get_order_simulator(
        self, instrument_code, is_subsystem: bool
    ) -> MinuteOrderSimulatorOfLimitOrders:
        order_simulator = MinuteOrderSimulatorOfLimitOrders(
            system_accounts_stage=self,
            instrument_code=instrument_code,
            is_subsystem=is_subsystem,
        )
        return order_simulator


from copy import copy

import numpy as np
import pandas as pd

from syscore.pandas.full_merge_with_replacement import full_merge_of_existing_series
from sysobjects.dict_of_named_futures_per_contract_prices import (
    contract_name_from_column_name,
)
from sysobjects.futures_per_contract_prices import (
    PRICE_DATA_COLUMNS,
    FINAL_COLUMN,
    VOLUME_COLUMN,
    NOT_VOLUME_COLUMNS,
)
from sysobjects.dict_of_futures_per_contract_prices import dictFuturesContractPrices
from sysobjects.roll_calendars import rollCalendar


class futuresAdjustedPrices(pd.DataFrame):
    """
    adjusted price information
    """

    def __init__(self, price_data):
        price_data.index.name = "index"  # arctic compatible
        super().__init__(price_data)

    @classmethod
    def create_empty(futuresContractPrices):
        """
        Our graceful fail is to return an empty, but valid, dataframe
        """

        futures_contract_prices = futuresContractPrices(pd.Series(dtype="float64"))

        return futures_contract_prices

    @classmethod
    def stitch_individual_contracts_from_roll_calendars(
        futuresAdjustedPrices,
        individual_contracts: dictFuturesContractPrices,
        roll_calendar: rollCalendar,
    ):
        """
        Do backstitching of multiple prices using panama method

        If you want to change then override this method

        :param individual_contracts: dict of individual futures prices
        :param roll_calendar: rollCalendar
        :param forward_fill: forward fill prices and forwards before stitching

        :return: futuresAdjustedPrices

        """
        adjusted_prices = _panama_stitch(individual_contracts, roll_calendar)
        return futuresAdjustedPrices(adjusted_prices)

    def update_with_individual_contract_prices_no_roll(
        self,
        individual_contracts: dictFuturesContractPrices,
        roll_calendar: rollCalendar,
    ):
        """
        Update adjusted prices assuming no roll has happened

        :param individual_contracts: dictFuturesContractPrices
        :param roll_calendar: rollCalendar
        :return: updated adjusted prices
        """

        updated_adj = _update_adjusted_prices_from_individual_contracts_no_roll(
            self, individual_contracts, roll_calendar
        )

        return updated_adj


def _panama_stitch(
    individual_contracts: dictFuturesContractPrices,
    roll_calendar: rollCalendar,
) -> pd.DataFrame:
    """
    Do a panama stitch for adjusted prices

    :param individual_contracts:  dictFuturesContractPrices
    :return: pd.Series of adjusted prices
    """
    individual_contracts = copy(individual_contracts)

    if individual_contracts.empty:
        raise Exception("Can't stitch an empty dictFuturesContractPrices object")

    previous_calendar = roll_calendar.iloc[0]
    previous_contract_prices = individual_contracts[str(previous_calendar.current_contract)]
    previous_row = previous_contract_prices.iloc[0:, ]
    adjusted_prices_values = [previous_row[PRICE_DATA_COLUMNS]]

    roll_calendar = roll_calendar.tz_localize(previous_contract_prices.index[0].tz)

    for dateindex in previous_contract_prices.loc[(previous_contract_prices.index == previous_calendar.name):].index:
        current_calendar = roll_calendar.reindex(previous_contract_prices.index, method='ffill').loc[dateindex, :]
        current_contract_prices = individual_contracts[str(current_calendar.current_contract)]
        current_row = current_contract_prices.loc[dateindex]

        if current_calendar.current_contract == previous_calendar.current_contract:
            # no roll has ocurred
            # we just append the price
            adjusted_prices_values.append(current_row[PRICE_DATA_COLUMNS])
        else:
            # A roll has occured:
            adjusted_prices_values = _roll_in_panama(
                adjusted_prices_values,
                str(previous_calendar.current_contract),
                previous_row,
                str(current_calendar.current_contract),
                current_row,
            )

        previous_calendar = current_calendar
        previous_contract_prices = current_contract_prices
        previous_row = current_row

    # it's ok to return a DataFrame since the calling object will change the
    # type
    adjusted_prices = pd.DataFrame(adjusted_prices_values)

    return adjusted_prices


def _roll_in_panama(adjusted_prices_values, previous_contract, previous_row, current_contract, current_row):
    # This is the sort of code you will need to change to adjust the roll logic
    # The roll differential is from the previous_row
    roll_differential = previous_row.FINAL - previous_row.FINAL
    if np.isnan(roll_differential):
        raise Exception(
            "On this day %s which should be a roll date we don't have prices for both %s and %s contracts"
            % (
                str(current_row.name),
                previous_contract,
                current_contract,
            )
        )

    # We add the roll differential to all previous prices
    volume = [row[VOLUME_COLUMN] for row in adjusted_prices_values]
    adjusted_prices_values = [
        adj_price + roll_differential for adj_price in adjusted_prices_values[NOT_VOLUME_COLUMNS]
    ]
    adjusted_prices_values = [
        pd.concat([adjusted_prices_values[i], volume[i]], axis=1) for i in range(len(adjusted_prices_values))
    ]
    # note this includes the price for the previous row, which will now be equal to the forward price
    # We now add todays price. This will be for the new contract

    adjusted_prices_values.append(current_row[PRICE_DATA_COLUMNS])

    return adjusted_prices_values


no_update_roll_has_occured = futuresAdjustedPrices.create_empty()


def _update_adjusted_prices_from_individual_contracts_no_roll(
    existing_adjusted_prices: futuresAdjustedPrices,
    individual_contracts: dictFuturesContractPrices,
    roll_calendar: rollCalendar,
) -> futuresAdjustedPrices:
    """
    Update adjusted prices assuming no roll has happened

    :param existing_adjusted_prices: futuresAdjustedPrices
    :param individual_contracts: dictFuturesContractPrices
    :param roll_calendar: rollCalendar
    :return: updated adjusted prices
    """
    last_date_in_existing_adjusted_prices = existing_adjusted_prices.index[-1]
    last_contract_in_roll_calendar = roll_calendar.current_contract.iloc[-1]
    try:
        new_contract_price_data = individual_contracts[str(last_contract_in_roll_calendar)]
    except KeyError:
        raise Exception(
            "No contract named %s in dict of individual contracts" % (
                str(last_contract_in_roll_calendar)
            )
        )

    new_adjusted_prices = (
        new_contract_price_data.loc[new_contract_price_data.index > last_date_in_existing_adjusted_prices]
    )

    filled_roll_calendar = roll_calendar.reindex_like(new_contract_price_data, method='ffill')
    no_roll_has_occurred = (
        last_contract_in_roll_calendar == filled_roll_calendar.loc[
            filled_roll_calendar.index < last_date_in_existing_adjusted_prices
        ].tail(1).current_contract
    )

    if not no_roll_has_occurred:
        return no_update_roll_has_occured

    new_adjusted_prices.dropna(inplace=True)

    merged_adjusted_prices = full_merge_of_existing_series(
        existing_adjusted_prices, new_adjusted_prices
    )
    merged_adjusted_prices = futuresAdjustedPrices(merged_adjusted_prices)

    return merged_adjusted_prices

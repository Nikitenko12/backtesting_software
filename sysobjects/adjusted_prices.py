import datetime
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
from sysobjects.sessions import Session


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
        sessions: Session,
        backadjust: bool=True,
    ):
        """
        Do backstitching of multiple prices using panama method

        If you want to change then override this method

        :param individual_contracts: dict of individual futures prices
        :param roll_calendar: rollCalendar
        :param forward_fill: forward fill prices and forwards before stitching

        :return: futuresAdjustedPrices

        """
        adjusted_prices = _panama_stitch_vectorized(individual_contracts, roll_calendar, sessions, backadjust)
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


def _panama_stitch_vectorized(
    individual_contracts: dictFuturesContractPrices,
    roll_calendar: rollCalendar,
    sessions: Session,
    backadjust: bool=True,
) -> pd.DataFrame:
    individual_contracts = copy(individual_contracts)
    roll_calendar = roll_calendar.tz_localize(sessions.tzinfo)

    if individual_contracts.empty:
        raise Exception("Can't stitch an empty dictFuturesContractPrices object")

    initial_row = roll_calendar.iloc[0]
    initial_contract = individual_contracts[str(initial_row.current_contract)].tz_localize('utc').tz_convert(sessions.tzinfo)
    current_contract = [initial_contract.loc[:initial_contract.index.to_series().asof(initial_row.name)]]
    roll_differential = pd.Series(dtype=float, index=roll_calendar.index)
    roll_differential.iloc[0] = 0.0

    for roll_dt, row in roll_calendar.iterrows():
        if sessions.start_time > sessions.end_time:
            roll_date = roll_dt.date() - pd.Timedelta(days=1)
        else:
            roll_date = roll_dt.date()
        session_start_dt_for_roll = pd.Timestamp(f'{roll_date} {sessions.start_time}', tzinfo=sessions.tzinfo)
        next_contract = individual_contracts[str(row.next_contract)].tz_localize('utc').tz_convert(sessions.tzinfo)

        actual_roll_dt = next_contract.index.to_series().asof(session_start_dt_for_roll)
        current_contract.append(
            next_contract.loc[actual_roll_dt:]
        )

        actual_roll_dt_for_previous_contract = current_contract[-2].index.to_series().asof(session_start_dt_for_roll)
        roll_differential.loc[session_start_dt_for_roll] = (
            current_contract[-1].loc[actual_roll_dt, 'FINAL'] - current_contract[-2].loc[actual_roll_dt_for_previous_contract, 'FINAL']
        )
        current_contract[-2] = current_contract[-2].loc[:actual_roll_dt_for_previous_contract]

        roll_differential.drop(index=roll_dt, inplace=True)

    current_contract = pd.concat(current_contract, axis=0).sort_index()
    current_contract['index'] = current_contract.index
    current_contract = current_contract.drop_duplicates(subset='index', keep='last').drop(columns=['index'])

    if backadjust:
        price_index_series = current_contract.index.to_series()
        for roll_dt, roll_diff in roll_differential.items():
            actual_roll_dt = price_index_series.asof(roll_dt)
            current_contract.loc[:actual_roll_dt, NOT_VOLUME_COLUMNS] += roll_diff

    return current_contract


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
    roll_calendar = roll_calendar.tz_localize(individual_contracts[list(individual_contracts.keys())[0]].index[0].tz)

    if individual_contracts.empty:
        raise Exception("Can't stitch an empty dictFuturesContractPrices object")

    previous_calendar = roll_calendar.iloc[0]
    previous_contract_prices = individual_contracts[str(previous_calendar.current_contract)]
    previous_row = previous_contract_prices.iloc[0]
    adjusted_prices_values = [previous_row[PRICE_DATA_COLUMNS]]

    valid_indexes = pd.concat(list(individual_contracts.values()), axis=0).sort_index().index.drop_duplicates()
    complete_roll_calendar = roll_calendar.reindex(valid_indexes, method='ffill').bfill().astype(int)

    for dateindex in list(valid_indexes)[1:]:
        current_calendar = complete_roll_calendar.loc[dateindex, :]
        current_contract_prices = individual_contracts[str(current_calendar.current_contract)]
        print(current_calendar)

        if dateindex in current_contract_prices.index:
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

            previous_calendar = copy(current_calendar)
            previous_contract_prices = copy(current_contract_prices)
            previous_row = copy(current_row)

    # it's ok to return a DataFrame since the calling object will change the
    # type
    adjusted_prices = pd.DataFrame(adjusted_prices_values)

    return adjusted_prices


def _roll_in_panama(adjusted_prices_values, previous_contract, previous_row, current_contract, current_row):
    # This is the sort of code you will need to change to adjust the roll logic
    # The roll differential is from the previous_row
    roll_differential = current_row.FINAL - previous_row.FINAL
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
    # volume = [row[VOLUME_COLUMN] for row in adjusted_prices_values]
    for i in range(len(adjusted_prices_values)):
        adjusted_prices_values[i][NOT_VOLUME_COLUMNS] += roll_differential

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

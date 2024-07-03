"""
We create adjusted prices using multiple prices stored in database

We then store those adjusted prices in database and/or csv

"""
from syscore.constants import arg_not_supplied
from syscore.dateutils import MINUTE_FREQ
from sysdata.csv.csv_adjusted_prices import csvFuturesAdjustedPricesData
from sysdata.csv.csv_roll_calendars import csvRollCalendarData
from sysdata.futures.roll_calendars import rollCalendarData

from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysobjects.roll_calendars import rollCalendar

from sysproduction.data.prices import diagPrices

diag_prices = diagPrices()


def _get_data_inputs(csv_adj_data_path, csv_roll_calendar_data_path):
    db_individual_contract_prices = diag_prices.db_futures_contract_price_data
    db_adjusted_prices = diag_prices.db_futures_adjusted_prices_data
    csv_adjusted_prices = csvFuturesAdjustedPricesData(csv_adj_data_path)
    roll_calendar_data = csvRollCalendarData(csv_roll_calendar_data_path)

    return db_individual_contract_prices, db_adjusted_prices, csv_adjusted_prices, roll_calendar_data


def process_adjusted_prices_all_instruments(
    csv_adj_data_path=arg_not_supplied, csv_roll_calendar_data_path=arg_not_supplied, ADD_TO_DB=True,
):
    db_individual_contract_prices, _notused, _alsonotused, _notusedaswell = _get_data_inputs(csv_adj_data_path, csv_roll_calendar_data_path)
    instrument_list = db_individual_contract_prices.get_list_of_instrument_codes_with_price_data_at_frequency(MINUTE_FREQ)
    for instrument_code in instrument_list:
        print(instrument_code)
        process_adjusted_prices_single_instrument(
            instrument_code,
            csv_adj_data_path=csv_adj_data_path,
            csv_roll_calendar_data_path=csv_roll_calendar_data_path,
            ADD_TO_DB=ADD_TO_DB,
        )


def process_adjusted_prices_single_instrument(
    instrument_code,
    csv_adj_data_path=arg_not_supplied,
    csv_roll_calendar_data_path=arg_not_supplied,
    ADD_TO_DB=True,
):
    (
        influx_individual_contract_prices,
        parquet_adjusted_prices,
        csv_adjusted_prices,
        csv_roll_calendar_data,
    ) = _get_data_inputs(csv_adj_data_path, csv_roll_calendar_data_path)

    roll_calendar = csv_roll_calendar_data.get_roll_calendar(instrument_code)
    influx_individual_contract_prices = influx_individual_contract_prices.get_prices_at_frequency_for_instrument(instrument_code, MINUTE_FREQ)

    roll_calendar = roll_calendar.iloc[-2:]

    adjusted_prices = futuresAdjustedPrices.stitch_individual_contracts_from_roll_calendars(
        influx_individual_contract_prices, roll_calendar
    )

    print(adjusted_prices)

    if ADD_TO_DB:
        parquet_adjusted_prices.add_adjusted_prices(
            instrument_code, adjusted_prices, ignore_duplication=True
        )

    return adjusted_prices


if __name__ == "__main__":
    input("Will overwrite existing prices are you sure?! CTL-C to abort")
    # modify flags and datapath as required
    process_adjusted_prices_all_instruments(
        csv_adj_data_path=arg_not_supplied, csv_roll_calendar_data_path=arg_not_supplied, ADD_TO_DB=True,
    )

from sysdata.futures.adjusted_prices import (
    futuresAdjustedPricesData,
)
from sysobjects.adjusted_prices import futuresAdjustedPrices
from sysdata.influxdb.influx_connection import influxData
from syslogging.logger import *
import pandas as pd

ADJPRICE_BUCKET = "futures_adjusted_prices"


class influxFuturesAdjustedPricesData(futuresAdjustedPricesData):
    """
    Class to read / write multiple futures price data to and from arctic
    """

    def __init__(self, influx_db=None, log=get_logger("influxFuturesAdjustedPrices")):
        super().__init__(log=log)

        self._influx = influxData(ADJPRICE_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx)

    @property
    def influx(self):
        return self._influx

    def get_list_of_instruments(self) -> list:
        return self.influx.get_keynames()

    def _get_adjusted_prices_without_checking(
        self, instrument_code: str
    ) -> futuresAdjustedPrices:
        data = self.influx.read(instrument_code)

        instrpricedata = futuresAdjustedPrices(data)

        return instrpricedata

    def _delete_adjusted_prices_without_any_warning_be_careful(
        self, instrument_code: str
    ):
        self.influx.delete(instrument_code)
        self.log.debug(
            "Deleted adjusted prices for %s from %s" % (instrument_code, str(self)),
            instrument_code=instrument_code,
        )

    def _add_adjusted_prices_without_checking_for_existing_entry(
        self, instrument_code: str, adjusted_price_data: futuresAdjustedPrices
    ):
        adjusted_price_data_aspd = pd.DataFrame(adjusted_price_data)
        adjusted_price_data_aspd = adjusted_price_data_aspd.astype(float)
        self.influx.write(instrument_code, adjusted_price_data_aspd)
        self.log.debug(
            "Wrote %s lines of prices for %s to %s"
            % (len(adjusted_price_data), instrument_code, str(self)),
            instrument_code=instrument_code,
        )

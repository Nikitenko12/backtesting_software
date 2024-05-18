from sysdata.fx.spotfx import fxPricesData
from sysobjects.spot_fx_prices import fxPrices
from sysdata.influxdb.influx_connection import influxData
from syslogging.logger import *
import pandas as pd

SPOTFX_BUCKET = "spotfx_prices"


class influxFxPricesData(fxPricesData):
    """
    Class to read / write fx prices
    """

    def __init__(self, influx_db=None, log=get_logger("influxFxPricesData")):
        super().__init__(log=log)
        self._influx = influxData(SPOTFX_BUCKET, influx_db=influx_db)

    @property
    def influx(self):
        return self._influx

    def __repr__(self):
        return repr(self._influx)

    def get_list_of_fxcodes(self) -> list:
        return self.influx.get_keynames()

    def _get_fx_prices_without_checking(self, currency_code: str) -> fxPrices:
        fx_data = self.influx.read(currency_code)

        fx_prices = fxPrices(fx_data[fx_data.columns[0]])

        return fx_prices

    def _delete_fx_prices_without_any_warning_be_careful(self, currency_code: str):
        self.influx.delete(currency_code)
        self.log.debug(
            "Deleted fX prices for %s from %s" % (currency_code, str(self)),
            **{CURRENCY_CODE_LOG_LABEL: currency_code, "method": "temp"},
        )

    def _add_fx_prices_without_checking_for_existing_entry(
        self, currency_code: str, fx_price_data: fxPrices
    ):
        fx_price_data_aspd = pd.DataFrame(fx_price_data)
        fx_price_data_aspd.columns = ["price"]
        fx_price_data_aspd = fx_price_data_aspd.astype(float)

        self.influx.write(currency_code, fx_price_data_aspd)
        self.log.debug(
            "Wrote %s lines of prices for %s to %s"
            % (len(fx_price_data), currency_code, str(self)),
            **{CURRENCY_CODE_LOG_LABEL: currency_code, "method": "temp"},
        )

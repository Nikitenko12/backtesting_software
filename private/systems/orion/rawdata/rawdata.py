from systems.rawdata import RawData


import pandas as pd

from systems.system_cache import diagnostic, output
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR


class OrionRawData(RawData):
    def get_aggregated_minute_prices(self, instrument_code: str, barsize: str = '5T'):
        minuteprice = self.get_minute_prices(instrument_code)
        agg_minuteprice = minuteprice.resample(barsize).agg(
            {
                'OPEN': 'first',
                'HIGH': 'max',
                'LOW': 'min',
                'LAST': 'last',
                'VOLUME': 'sum',
            }
        )

        return agg_minuteprice

    def get_minute_prices(self, instrument_code: str):
        self.log.debug(
            "Calculating minute prices for %s" % instrument_code,
            instrument_code=instrument_code,
        )
        minuteprice = self.data_stage.minute_prices(instrument_code)

        if len(minuteprice) == 0:
            raise Exception(
                "Data for %s not found! Remove from instrument list, or add to config.ignore_instruments"
                % instrument_code
            )

        return minuteprice



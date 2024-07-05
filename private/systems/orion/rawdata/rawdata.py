from systems.stage import SystemStage
from sysdata.config.configdata import Config
from sysdata.sim.futures_sim_data import futuresSimData

import pandas as pd

from systems.system_cache import diagnostic, output
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR


class OrionRawData(SystemStage):
    def get_aggregated_minute_prices(self, instrument_code: str, barsize: str = '5T'):
        minuteprice = self.get_minute_prices(instrument_code)
        agg_minuteprice = minuteprice.resample(barsize).agg(
            {
                'OPEN': 'first',
                'HIGH': 'max',
                'LOW': 'min',
                'FINAL': 'last',
                'VOLUME': 'sum',
            }
        )

        return agg_minuteprice

    def get_minute_prices(self, instrument_code: str):
        self.log.debug(
            "Calculating minute prices for %s" % instrument_code,
            instrument_code=instrument_code,
        )
        minuteprice = self.data_stage.get_backadjusted_futures_price(instrument_code)

        if len(minuteprice) == 0:
            raise Exception(
                "Data for %s not found! Remove from instrument list, or add to config.ignore_instruments"
                % instrument_code
            )

        minuteprice = minuteprice.loc[minuteprice['FINAL'] != 0.0]
        return minuteprice

    @property
    def name(self):
        return "rawdata"

    @property
    def data_stage(self) -> futuresSimData:
        return self.parent.data

    @property
    def config(self) -> Config:
        return self.parent.config

    def get_raw_cost_data(self, instrument_code: str):
        return self.data_stage.get_raw_cost_data(instrument_code)

    def get_value_of_block_price_move(self, instrument_code: str) -> float:
        return self.data_stage.get_value_of_block_price_move(instrument_code)

    def get_fx_for_instrument(self, instrument_code: str, base_currency: str):
        return self.data_stage.get_fx_for_instrument(
            instrument_code=instrument_code, base_currency=base_currency
        )

    @output()
    def daily_denominator_price(self, instrument_code: str) -> pd.DataFrame:
        """
        Gets daily prices for use with % volatility
        This won't always be the same as the normal 'price'

        :param instrument_code: Instrument to get prices for
        :type trading_rules: str

        :returns: Tx1 pd.DataFrame

        """
        self.log.warning(
            "No carry data found for %s, using adjusted prices to calculate percentage returns"
            % instrument_code
        )
        return self.get_minute_prices(instrument_code)

    # @input
    # def get_daily_prices(self, instrument_code) -> pd.DataFrame:
    #     """
    #     Gets daily prices
    #
    #     :param instrument_code: Instrument to get prices for
    #     :type trading_rules: str
    #
    #     :returns: Tx1 pd.DataFrame
    #
    #     KEY OUTPUT
    #     """
    #     self.log.debug(
    #         "Calculating daily prices for %s" % instrument_code,
    #         instrument_code=instrument_code,
    #     )
    #     dailyprice = self.data_stage.daily_prices(instrument_code)
    #
    #     if len(dailyprice) == 0:
    #         raise Exception(
    #             "Data for %s not found! Remove from instrument list, or add to config.ignore_instruments"
    #             % instrument_code
    #         )
    #
    #     return dailyprice
    #
    # @input
    # def get_natural_frequency_prices(self, instrument_code: str) -> pd.DataFrame:
    #     self.log.debug(
    #         "Retrieving natural prices for %s" % instrument_code,
    #         instrument_code=instrument_code,
    #     )
    #
    #     natural_prices = self.data_stage.get_raw_price(instrument_code)
    #
    #     if len(natural_prices) == 0:
    #         raise Exception(
    #             "Data for %s not found! Remove from instrument list, or add to config.ignore_instruments"
    #         )
    #
    #     return natural_prices

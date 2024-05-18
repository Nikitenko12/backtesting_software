from syscore.exceptions import missingData
from sysdata.production.capital import capitalData

CAPITAL_BUCKET = "influx_capital"

from sysdata.influxdb.influx_connection import influxData
from syslogging.logger import *
import pandas as pd


class influxCapitalData(capitalData):
    """
    Class to read / write multiple total capital data to and from influx
    """

    def __init__(self, influx_db=None, log=get_logger("influxCapitalData")):
        super().__init__(log=log)

        self._influx = influxData(CAPITAL_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx)

    @property
    def influx(self):
        return self._influx

    def _get_list_of_strategies_with_capital_including_total(self) -> list:
        return self.influx.get_keynames()

    def get_capital_pd_df_for_strategy(self, strategy_name: str) -> pd.DataFrame:
        try:
            pd_series = self.influx.read(strategy_name)
        except:
            raise missingData(
                "Unable to get capital data from influx for strategy %s" % strategy_name
            )

        return pd_series

    def _delete_all_capital_for_strategy_no_checking(self, strategy_name: str):
        self.influx.delete(strategy_name)

    def update_capital_pd_df_for_strategy(
        self, strategy_name: str, updated_capital_df: pd.DataFrame
    ):
        self.influx.write(strategy_name, updated_capital_df)

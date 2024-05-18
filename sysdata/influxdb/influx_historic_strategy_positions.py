import datetime
import pandas as pd

from sysobjects.production.tradeable_object import (
    listOfInstrumentStrategies,
    instrumentStrategy,
)
from sysdata.influxdb.influx_connection import influxData
from sysdata.production.historic_strategy_positions import strategyPositionData
from syscore.exceptions import missingData

from syslogging.logger import *

STRATEGY_POSITION_BUCKET = "strategy_positions"


class influxStrategyPositionData(strategyPositionData):
    def __init__(self, influx_db=None, log=get_logger("influxStrategyPositionData")):
        super().__init__(log=log)

        self._influx = influxData(STRATEGY_POSITION_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx)

    @property
    def influx(self):
        return self._influx

    def get_list_of_instrument_strategies(self) -> listOfInstrumentStrategies:
        list_of_keys = self.influx.get_keynames()
        list_of_instrument_strategies = [
            instrumentStrategy.from_key(key) for key in list_of_keys
        ]

        return listOfInstrumentStrategies(list_of_instrument_strategies)

    def _write_updated_position_series_for_instrument_strategy_object(
        self, instrument_strategy: instrumentStrategy, updated_series: pd.Series
    ):
        ident = instrument_strategy.key
        updated_data_as_df = pd.DataFrame(updated_series)
        updated_data_as_df.columns = ["position"]

        self.influx.write(ident=ident, data=updated_data_as_df)

    def _delete_position_series_for_instrument_strategy_object_without_checking(
        self, instrument_strategy: instrumentStrategy
    ):
        ident = instrument_strategy.key
        self.influx.delete(ident)

    def get_position_as_series_for_instrument_strategy_object(
        self, instrument_strategy: instrumentStrategy
    ) -> pd.Series:
        keyname = instrument_strategy.key
        try:
            pd_df = self.influx.read(keyname)
        except:
            raise missingData

        return pd_df.iloc[:, 0]

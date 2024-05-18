"""
Read and write data from mongodb for individual futures contracts

"""
from typing import Union

from syscore.exceptions import missingData
from sysdata.influxdb.influx_connection import influxData
from sysdata.production.optimal_positions import optimalPositionData
from syslogging.logger import *

from sysobjects.production.tradeable_object import (
    instrumentStrategy,
    listOfInstrumentStrategies,
)

import pandas as pd

OPTIMAL_POSITION_BUCKET = "optimal_positions"


class influxOptimalPositionData(optimalPositionData):
    def __init__(self, influx_db=None, log=get_logger("influxOptimalPositionData")):
        super().__init__(log=log)

        self._influx_connection = influxData(
            OPTIMAL_POSITION_BUCKET,
            influx_db=influx_db
        )

    def __repr__(self):
        return repr(self._influx_connection)

    @property
    def influx_connection(self):
        return self._influx_connection

    def get_list_of_instrument_strategies_with_optimal_position(
        self,
    ) -> listOfInstrumentStrategies:
        raw_list_of_instrument_strategies = self.influx_connection.get_keynames()
        list_of_instrument_strategies = [
            instrumentStrategy.from_key(key)
            for key in raw_list_of_instrument_strategies
        ]

        return listOfInstrumentStrategies(list_of_instrument_strategies)

    def get_optimal_position_as_df_for_instrument_strategy(
        self, instrument_strategy: instrumentStrategy
    ) -> pd.DataFrame:
        try:
            ident = instrument_strategy.key
            df_result = self.influx_connection.read(ident)
        except:
            raise missingData

        return df_result

    def write_optimal_position_as_df_for_instrument_strategy_without_checking(
        self,
        instrument_strategy: instrumentStrategy,
        optimal_positions_as_df: pd.DataFrame,
    ):
        ident = instrument_strategy.key
        self.influx_connection.write(ident, optimal_positions_as_df)

import datetime
import pandas as pd

from sysobjects.contracts import futuresContract, listOfFuturesContracts
from sysdata.influxdb.influx_connection import influxData
from sysdata.production.historic_contract_positions import contractPositionData

from syscore.exceptions import missingData

from syslogging.logger import *

CONTRACT_POSITION_BUCKET = "contract_positions"


class influxContractPositionData(contractPositionData):
    def __init__(self, influx_db=None, log=get_logger("influxContractPositionData")):
        super().__init__(log=log)

        self._influx = influxData(CONTRACT_POSITION_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx)

    @property
    def influx(self):
        return self._influx

    def _write_updated_position_series_for_contract_object(
        self, contract_object: futuresContract, updated_series: pd.Series
    ):
        ## overwrites what is there without checking
        ident = contract_object.key
        updated_data_as_df = pd.DataFrame(updated_series)
        updated_data_as_df.columns = ["position"]

        self.influx.write(ident=ident, data=updated_data_as_df)

    def _delete_position_series_for_contract_object_without_checking(
        self, contract_object: futuresContract
    ):
        ident = contract_object.key
        self.influx.delete(ident)

    def get_position_as_series_for_contract_object(
        self, contract_object: futuresContract
    ) -> pd.Series:
        keyname = contract_object.key
        try:
            pd_df = self.influx.read(keyname)
        except:
            raise missingData

        return pd_df.iloc[:, 0]

    def get_list_of_contracts(self) -> listOfFuturesContracts:
        ## doesn't remove zero positions
        list_of_keys = self.influx.get_keynames()
        list_of_futures_contract = [
            futuresContract.from_key(key) for key in list_of_keys
        ]

        return listOfFuturesContracts(list_of_futures_contract)

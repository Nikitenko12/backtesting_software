from sysdata.futures.spreads import spreadsForInstrumentData
from sysobjects.spreads import spreadsForInstrument
from sysdata.influxdb.influx_connection import influxData
from syslogging.logger import *
import pandas as pd

SPREAD_BUCKET = "spreads"
SPREAD_COLUMN_NAME = "spread"


class influxSpreadsForInstrumentData(spreadsForInstrumentData):
    def __init__(self, influx_db=None, log=get_logger("influxSpreadsForInstrument")):
        super().__init__(log=log)

        self._influx = influxData(SPREAD_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx)

    @property
    def influx(self):
        return self._influx

    def get_list_of_instruments(self) -> list:
        return self.influx.get_keynames()

    def _get_spreads_without_checking(
        self, instrument_code: str
    ) -> spreadsForInstrument:
        data = self.influx.read(instrument_code)

        spreads = spreadsForInstrument(data[SPREAD_COLUMN_NAME])

        return spreads

    def _delete_spreads_without_any_warning_be_careful(self, instrument_code: str):
        self.influx.delete(instrument_code)
        self.log.debug(
            "Deleted spreads for %s from %s" % (instrument_code, str(self)),
            instrument_code=instrument_code,
        )

    def _add_spreads_without_checking_for_existing_entry(
        self, instrument_code: str, spreads: spreadsForInstrument
    ):
        spreads_as_pd = pd.DataFrame(spreads)
        spreads_as_pd.columns = [SPREAD_COLUMN_NAME]
        spreads_as_pd = spreads_as_pd.astype(float)
        self.influx.write(instrument_code, spreads_as_pd)
        self.log.debug(
            "Wrote %s lines of spreads for %s to %s"
            % (len(spreads_as_pd), instrument_code, str(self)),
            instrument_code=instrument_code,
        )

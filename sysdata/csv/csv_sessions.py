from syscore.constants import arg_not_supplied
from syscore.fileutils import resolve_path_and_filename_for_package
from sysdata.futures.sessions import sessionsData
from sysobjects.sessions import Session
from syslogging.logger import *

import pandas as pd
from datetime import time
import pytz

SESSIONS_DATAPATH = "data.futures.csvconfig"
SESSIONS_CONFIG_FILE = "sessions.csv"


class allSessions(pd.DataFrame):
    @classmethod
    def read_from_file(allSessions, filename):
        try:
            sessions_data = pd.read_csv(filename)
        except BaseException:
            raise Exception("Can't read file %s" % filename)

        try:
            sessions_data.index = sessions_data.Instrument
            sessions_data.drop(labels="Instrument", axis=1, inplace=True)

            sessions_data['SessionStartTime'] = (
                [time.fromisoformat(start_time) for start_time in sessions_data['SessionStartTime']]
            )
            sessions_data['SessionEndTime'] = (
                [time.fromisoformat(end_time) for end_time in sessions_data['SessionEndTime']]
            )
            sessions_data['Timezone'] = (
                pytz.timezone(tz) for tz in sessions_data['Timezone']
            )

        except BaseException:
            raise Exception("Badly configured file %s" % (filename))

        return allSessions(sessions_data)

    def get_list_of_instruments(self) -> list:
        return list(self.index)

    def get_sessions_for_instrument(
        self, instrument_code: str
    ) -> Session:
        config_for_this_instrument = self.loc[instrument_code]
        sessions_object = Session(
            start_time=config_for_this_instrument.SessionStartTime,
            end_time=config_for_this_instrument.SessionEndTime,
            tz=config_for_this_instrument.Timezone
        )

        return sessions_object

    def update_sessions_for_instrument(
        self, instrument_code: str, session: Session
    ):
        self.at[instrument_code, "SessionStartTime"] = session.start_time
        self.at[instrument_code, "SessionEndTime"] = session.end_time
        self.at[instrument_code, "Timezone"] = session.tzinfo

    def write_to_file(self, filename: str):
        self.to_csv(filename, index_label="Instrument")


class csvSessionsData(sessionsData):
    def __init__(
        self, log=get_logger("csvSessionsData"), datapath=arg_not_supplied
    ):
        super().__init__(log=log)
        if datapath is arg_not_supplied:
            datapath = SESSIONS_DATAPATH
        config_file = resolve_path_and_filename_for_package(datapath, SESSIONS_CONFIG_FILE)

        self._config_file = config_file

    def __repr__(self):
        return "Sessions data for initialising system config"

    def get_list_of_instruments(self) -> list:
        all_sessions = self.get_sessions_for_all_instruments()
        return all_sessions.get_list_of_instruments()

    def _get_sessions_without_checking(
        self, instrument_code: str
    ) -> Session:
        all_parameters = self.get_sessions_for_all_instruments()
        return all_parameters.get_sessions_for_instrument(instrument_code)

    def _delete_sessions_data_without_any_warning_be_careful(
        self, instrument_code: str
    ):
        raise NotImplementedError("csv is read only")

    def _add_sessions_without_checking_for_existing_entry(
        self, instrument_code: str, sessions: Session
    ):
        ## We don't normally allow this, but a special case as we safe modify
        all_parameters = self.get_sessions_for_all_instruments()
        all_parameters.update_sessions_for_instrument(
            instrument_code, sessions
        )
        all_parameters.write_to_file(self.config_file)

        self.log.warning(
            "*** WRITTEN NEW SESSIONS TO %s - copy to /data/futures/csvconfig/sessions.csv NOW ***"
            % self.config_file
        )

    def write_all_sessions_data(self, sessions_df: pd.DataFrame):
        all_sessions = allSessions(sessions_df)
        all_sessions.write_to_file(self.config_file)

    def get_sessions_for_all_instruments(self) -> allSessions:
        return allSessions.read_from_file(self.config_file)

    @property
    def config_file(self):
        return self._config_file


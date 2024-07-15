import pandas as pd
import datetime


class Session(object):
    def __init__(self, start_time: datetime.time, end_time: datetime.time):

        self._start_time = start_time
        self._end_time = end_time

    def __repr__(self):
        return f'{self._start_time} to {self._end_time}'

    def __eq__(self, other):
        return self._start_time == other.start_time and self._end_time == other.end_time

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time


class dictOfSessions(dict):
    def __repr__(self):
        object_repr = "Dict of sessions with %s instruments" % len(
            self.keys()
        )
        return object_repr

    @classmethod
    def from_pd_df(cls, sessions_df: pd.DataFrame):
        assert "SessionStartTime" in sessions_df.columns
        assert "SessionEndTime" in sessions_df.columns

        dict_of_sessions = dict()
        for instrument_code, session in sessions_df.iterrows():
            assert isinstance(instrument_code, str)

            this_session = Session(
                datetime.time.fromisoformat(
                    session.SessionStartTime
                ),
                datetime.time.fromisoformat(
                    session.SessionEndTime
                ),
            )
            dict_of_sessions[instrument_code] = this_session

        return cls(dict_of_sessions)

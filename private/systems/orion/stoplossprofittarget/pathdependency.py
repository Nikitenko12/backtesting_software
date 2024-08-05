import pandas as pd
import numpy as np

from sysobjects.sessions import Session
from systems.stage import SystemStage
from systems.system_cache import input, diagnostic, output
from private.systems.orion.rawdata.rawdata import apply_sessions_to_aggregated_data
from copy import copy


class StopLossProfitTarget(SystemStage):
    """
    This stage goes directly after the rules stage
    """

    @property
    def name(self):
        return "pathdependency"

    def get_inputs(self, instrument_code: str):
        prices = self.get_prices(instrument_code)
        signals_dict = self.signals_dict(instrument_code)
        sessions = self.get_sessions(instrument_code)

        long_signals = signals_dict['long_signals']
        short_signals = signals_dict['short_signals']
        long_limit_prices = signals_dict['long_limit_prices']
        short_limit_prices = signals_dict['short_limit_prices']
        long_stop_loss_levels = signals_dict['long_stop_loss_prices']
        short_stop_loss_levels = signals_dict['short_stop_loss_prices']
        long_profit_target_levels = signals_dict['long_profit_taker']
        short_profit_target_levels = signals_dict['short_profit_taker']

        return (
            prices,
            sessions,
            long_signals,
            short_signals,
            long_limit_prices,
            short_limit_prices,
            long_stop_loss_levels,
            short_stop_loss_levels,
            long_profit_target_levels,
            short_profit_target_levels,
        )

    @output()
    def get_signals_after_limit_price_is_hit_stop_loss_and_profit_target(self, instrument_code: str):
        (
            prices,
            sessions,
            long_signals,
            short_signals,
            long_limit_prices,
            short_limit_prices,
            long_stop_loss_levels,
            short_stop_loss_levels,
            long_profit_target_levels,
            short_profit_target_levels,
        ) = self.get_inputs(instrument_code)

        signals_after_limit_prices_slpt = apply_limit_prices_slpt_to_signals(
            prices=prices,
            long_signals=long_signals,
            short_signals=short_signals,
            sessions=sessions,
            long_limit_prices=long_limit_prices,
            short_limit_prices=short_limit_prices,
            long_stop_loss_levels=long_stop_loss_levels,
            short_stop_loss_levels=short_stop_loss_levels,
            long_profit_target_levels=long_profit_target_levels,
            short_profit_target_levels=short_profit_target_levels,
        )

        return signals_after_limit_prices_slpt

    @property
    def prices(self) -> pd.DataFrame:
        return self.rules.data

    @property
    def rules(self):
        return self.parent.rules

    @input
    def signals_dict(self, instrument_code: str):
        return self.rules.get_raw_forecast(instrument_code, rule_variation_name='orion')

    def get_prices(self, instrument_code: str):
        return self.parent.rawdata.get_aggregated_minute_prices(
            instrument_code, barsize=self.parent.config.trading_rules['orion']['other_args']['small_timeframe']
        )

    def get_sessions(self, instrument_code: str):
        return self.parent.rawdata.get_sessions(instrument_code)


def apply_limit_prices_slpt_to_signals(
    prices: pd.DataFrame,
    sessions: Session,
    long_signals: pd.Series,
    short_signals: pd.Series,
    long_limit_prices: pd.Series,
    short_limit_prices: pd.Series,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
):
    assert prices.index.tz == sessions.tzinfo
    price_index_series = prices.index.to_series()

    signals = long_signals.astype(int) - short_signals.astype(int)
    where_signals_are_the_same = long_signals & short_signals

    long_limit_prices.loc[where_signals_are_the_same] = np.nan
    short_limit_prices.loc[where_signals_are_the_same] = np.nan
    long_stop_loss_levels.loc[where_signals_are_the_same] = np.nan
    short_stop_loss_levels.loc[where_signals_are_the_same] = np.nan
    long_profit_target_levels.loc[where_signals_are_the_same] = np.nan
    short_profit_target_levels.loc[where_signals_are_the_same] = np.nan

    trades = signals.loc[signals.ne(0)]
    it = iter(trades.items())

    path_dep_df = pd.DataFrame(
        columns=[
            'signals', 'dt_when_limit_price_was_hit',
            'dt_when_stop_loss_was_hit', 'dt_when_profit_target_was_hit',
            'dt_when_this_session_ended', 'dt_for_first_bar_in_next_session',
            'dt_when_trade_exited',
        ],
        index=trades.index,
    )

    for dt, signal in it:
        print(f'******* Signal {signal} generated at {dt} *******')
        datetime_starting_from_next_bar = price_index_series.mask(
            price_index_series <= dt, np.nan
        ).dropna()
        if len(datetime_starting_from_next_bar) == 0:
            break

        if signal > 0:
            limit_price = long_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'LOW'].le(limit_price)
            dt_when_limit_price_was_hit = pd.NaT if limit_price_was_hit.eq(
                False).all() else limit_price_was_hit.idxmax()

            stop_loss_level = long_stop_loss_levels[dt]
            profit_target_level = long_profit_target_levels[dt]

        else:
            limit_price = short_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(limit_price)
            dt_when_limit_price_was_hit = pd.NaT if limit_price_was_hit.eq(
                False).all() else limit_price_was_hit.idxmax()

            stop_loss_level = short_stop_loss_levels[dt]
            profit_target_level = short_profit_target_levels[dt]

        if dt_when_limit_price_was_hit is not pd.NaT:
            print(f'*** Limit price was hit at {dt_when_limit_price_was_hit}. Calculating exit times ***')
            datetime_starting_from_bar_when_limit_price_was_hit = price_index_series.mask(
                price_index_series < dt_when_limit_price_was_hit, np.nan
            ).dropna()
            if signal > 0:
                stop_loss_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'LOW'].le(
                    stop_loss_level)
                profit_target_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'HIGH'].ge(
                    profit_target_level)
            else:
                stop_loss_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'HIGH'].ge(
                    stop_loss_level)
                profit_target_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'LOW'].le(
                    profit_target_level)

            dt_when_stop_loss_was_hit = pd.NaT if stop_loss_was_hit.eq(False).all() else stop_loss_was_hit.idxmax()
            dt_when_profit_target_was_hit = pd.NaT if profit_target_was_hit.eq(False).all() else profit_target_was_hit.idxmax()

            date_when_this_session_ended = dt_when_limit_price_was_hit.date() if dt_when_limit_price_was_hit.time() < sessions.end_time or (
                sessions.end_time < sessions.start_time and dt_when_limit_price_was_hit.time() < sessions.start_time
            ) else (
                dt_when_limit_price_was_hit.date() + pd.Timedelta(1, 'D')
            )
            dt_when_this_session_ended = (
                pd.Timestamp(f'{date_when_this_session_ended} {sessions.end_time}').tz_localize(tz=sessions.tzinfo)
            )
            next_session = price_index_series.loc[price_index_series.asof(dt_when_this_session_ended):]
            dt_for_first_bar_in_next_session = pd.NaT if len(next_session) <= 1 else next_session.iloc[1]

        else:
            dt_when_stop_loss_was_hit = pd.NaT
            dt_when_profit_target_was_hit = pd.NaT
            dt_when_this_session_ended = pd.NaT
            dt_for_first_bar_in_next_session = pd.NaT

        when_trade_exited = pd.Series(
            [dt_when_stop_loss_was_hit, dt_when_profit_target_was_hit, dt_when_this_session_ended]
        ).dropna()
        dt_when_trade_exited = price_index_series.iloc[-1] if len(when_trade_exited) == 0 else when_trade_exited.min()

        path_dep_df.loc[dt, 'signals'] = signal
        path_dep_df.loc[dt, 'dt_when_limit_price_was_hit'] = dt_when_limit_price_was_hit
        path_dep_df.loc[dt, 'dt_when_stop_loss_was_hit'] = dt_when_stop_loss_was_hit
        path_dep_df.loc[dt, 'dt_when_profit_target_was_hit'] = dt_when_profit_target_was_hit
        path_dep_df.loc[dt, 'dt_when_this_session_ended'] = dt_when_this_session_ended
        path_dep_df.loc[dt, 'dt_for_first_bar_in_next_session'] = dt_for_first_bar_in_next_session
        path_dep_df.loc[dt, 'dt_when_trade_exited'] = dt_when_trade_exited

    print(f'****** Done calculating entry times for signals *******')
    trades_which_hit_limit_prices = path_dep_df.dt_when_limit_price_was_hit.dropna(inplace=False).index
    path_dep_df_after_limit_prices = path_dep_df.loc[trades_which_hit_limit_prices]

    previous_dt = path_dep_df_after_limit_prices.index[0]
    previous_row = path_dep_df_after_limit_prices.iloc[0]
    it = iter(path_dep_df_after_limit_prices.iloc[1:].iterrows())
    for dt, row in it:
        print(f'******* Checking signal {row.signals} generated at {dt} and hit at {row.dt_when_limit_price_was_hit} *******')
        if row.signals != previous_row.signals and dt <= previous_row.dt_when_limit_price_was_hit:
            print(f'*** Price hit opposing zone and generated opposite signal at {dt} before filling previous order at {previous_row.dt_when_limit_price_was_hit}. ' + (
                f'Cancelling previous order ***')
            )
            path_dep_df_after_limit_prices.drop(index=previous_dt, inplace=True)
            previous_dt = dt
            previous_row = row.copy()
        elif dt <= previous_row.dt_when_trade_exited:
            print(f'*** Signal generated at {dt} before previous order exited at {previous_row.dt_when_trade_exited}. Dropping signal ***')
            path_dep_df_after_limit_prices.drop(index=dt, inplace=True)
        else:
            print(f'*** Everything seems fine over here. Keeping signal ***')
            previous_dt = dt
            previous_row = row.copy()

    print(f'******* Done checking signals. There were {len(path_dep_df_after_limit_prices.index)} trades ' + (
        f'from {price_index_series.iloc[0]} to {price_index_series.iloc[-1]} *******')
    )
    next_session_if_exited_trade_at_eod = path_dep_df_after_limit_prices.loc[
        path_dep_df_after_limit_prices.dt_when_trade_exited == path_dep_df_after_limit_prices.dt_when_this_session_ended,
        'dt_for_first_bar_in_next_session'
    ].dropna()

    forecasts = pd.Series(np.nan, index=prices.index)
    forecasts[path_dep_df_after_limit_prices.dt_when_limit_price_was_hit] = signals[path_dep_df_after_limit_prices.index]
    forecasts[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    forecasts[next_session_if_exited_trade_at_eod] = 0
    forecasts = forecasts.ffill().fillna(0)

    new_long_limit_prices = pd.Series(np.nan, index=prices.index)
    new_long_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    new_long_limit_prices[
        price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited[path_dep_df_after_limit_prices.signals > 0])
    ] = 0
    new_long_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_long_limit_prices = new_long_limit_prices.ffill().replace(0, np.nan)

    new_short_limit_prices = pd.Series(np.nan, index=prices.index)
    new_short_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    new_short_limit_prices[
        price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited[path_dep_df_after_limit_prices.signals < 0])
    ] = 0
    new_short_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_short_limit_prices = new_short_limit_prices.ffill().replace(0, np.nan)

    stop_loss_levels = pd.Series(np.nan, index=prices.index)
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    stop_loss_levels[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    stop_loss_levels[next_session_if_exited_trade_at_eod] = 0
    stop_loss_levels = stop_loss_levels.ffill().replace(0, np.nan)

    profit_target_levels = pd.Series(np.nan, index=prices.index)
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    profit_target_levels[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    profit_target_levels[next_session_if_exited_trade_at_eod] = 0
    profit_target_levels = profit_target_levels.ffill().replace(0, np.nan)

    return dict(
        forecasts=forecasts,
        stop_loss_levels_after_slpt=stop_loss_levels,
        profit_target_levels_after_slpt=profit_target_levels,
        long_limit_prices_after_slpt=new_long_limit_prices,
        short_limit_prices_after_slpt=new_short_limit_prices,
        path_dep_df=path_dep_df_after_limit_prices,
    )

"""
def apply_limit_prices_slpt_to_signals(
    prices: pd.DataFrame,
    sessions: Session,
    long_limit_prices: pd.Series,
    short_limit_prices: pd.Series,
    signals: pd.Series,
    long_zones: pd.DataFrame,
    short_zones: pd.DataFrame,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
):
    assert prices.index.tz == sessions.tzinfo
    price_index_series = prices.index.to_series()

    trades = signals.loc[signals.ne(0)]
    it = iter(trades.items())

    path_dep_df = pd.DataFrame(
        columns=[
            'signals', 'dt_when_limit_price_was_hit', 'dt_when_zone_was_hit', 'dt_when_zone_changed',
            'dt_when_stop_loss_was_hit', 'dt_when_profit_target_was_hit',
            'dt_when_this_session_ended', 'dt_for_first_bar_in_next_session',
            'dt_when_trade_exited',
        ],
        index=trades.index,
    )

    for dt, signal in it:
        datetime_starting_from_next_bar = price_index_series.mask(
            price_index_series <= dt, np.nan
        ).dropna()
        if len(datetime_starting_from_next_bar) == 0:
            break

        if signal > 0:
            limit_price = long_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'LOW'].le(limit_price)
            dt_when_limit_price_was_hit = pd.NaT if limit_price_was_hit.eq(
                False).all() else limit_price_was_hit.idxmax()

            zone_to_be_hit = short_zones.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].cummin()
            zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].ge(
                zone_to_be_hit
            )

            zones_starting_from_this_bar = long_zones.mask(price_index_series < dt, np.nan).dropna()

            stop_loss_level = long_stop_loss_levels[dt]
            profit_target_level = long_profit_target_levels[dt]

        else:
            limit_price = short_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(limit_price)
            dt_when_limit_price_was_hit = pd.NaT if limit_price_was_hit.eq(
                False).all() else limit_price_was_hit.idxmax()

            zone_to_be_hit = long_zones.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].cummax()
            zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].le(
                zone_to_be_hit
            )

            zones_starting_from_this_bar = short_zones.mask(price_index_series < dt, np.nan).dropna()

            stop_loss_level = short_stop_loss_levels[dt]
            profit_target_level = short_profit_target_levels[dt]

        dt_when_zone_was_hit = pd.NaT if zone_was_hit.eq(False).all() else zone_was_hit.idxmax()

        zone_changes = zones_starting_from_this_bar.diff().iloc[1:].ne(0).any(axis=1)
        dt_when_zone_changed = pd.NaT if zone_changes.eq(False).all() else zone_changes.idxmax()

        if dt_when_limit_price_was_hit is not pd.NaT:
            datetime_starting_from_bar_when_limit_price_was_hit = price_index_series.mask(
                price_index_series <= dt_when_limit_price_was_hit, np.nan
            ).dropna()
            if signal > 0:
                stop_loss_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'LOW'].le(
                    stop_loss_level)
                profit_target_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'HIGH'].ge(
                    profit_target_level)
            else:
                stop_loss_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'HIGH'].ge(
                    stop_loss_level)
                profit_target_was_hit = prices.loc[datetime_starting_from_bar_when_limit_price_was_hit, 'LOW'].le(
                    profit_target_level)

            dt_when_stop_loss_was_hit = pd.NaT if stop_loss_was_hit.eq(False).all() else stop_loss_was_hit.idxmax()
            dt_when_profit_target_was_hit = pd.NaT if profit_target_was_hit.eq(False).all() else profit_target_was_hit.idxmax()

            date_when_this_session_ended = dt_when_limit_price_was_hit.date() if dt_when_limit_price_was_hit.time() < sessions.end_time or (
                sessions.end_time < sessions.start_time and dt_when_limit_price_was_hit.time() < sessions.start_time
            ) else (
                dt_when_limit_price_was_hit.date() + pd.Timedelta(1, 'D')
            )
            dt_when_this_session_ended = (
                pd.Timestamp(f'{date_when_this_session_ended} {sessions.end_time}').tz_localize(tz=price_index_series.index.tz)
            )
            next_session = price_index_series.loc[price_index_series.asof(dt_when_this_session_ended):]
            dt_for_first_bar_in_next_session = pd.NaT if len(next_session) <= 1 else next_session.iloc[1]

        else:
            dt_when_stop_loss_was_hit = pd.NaT
            dt_when_profit_target_was_hit = pd.NaT
            dt_when_this_session_ended = pd.NaT
            dt_for_first_bar_in_next_session = pd.NaT

        when_trade_exited = pd.Series(
            [dt_when_stop_loss_was_hit, dt_when_profit_target_was_hit, dt_when_this_session_ended]
        ).dropna()
        dt_when_trade_exited = pd.NaT if len(when_trade_exited) == 0 else when_trade_exited.min()

        path_dep_df.loc[dt, 'signals'] = signal
        path_dep_df.loc[dt, 'dt_when_limit_price_was_hit'] = dt_when_limit_price_was_hit
        path_dep_df.loc[dt, 'dt_when_zone_was_hit'] = dt_when_zone_was_hit
        path_dep_df.loc[dt, 'dt_when_zone_changed'] = dt_when_zone_changed
        path_dep_df.loc[dt, 'dt_when_stop_loss_was_hit'] = dt_when_stop_loss_was_hit
        path_dep_df.loc[dt, 'dt_when_profit_target_was_hit'] = dt_when_profit_target_was_hit
        path_dep_df.loc[dt, 'dt_when_this_session_ended'] = dt_when_this_session_ended
        path_dep_df.loc[dt, 'dt_for_first_bar_in_next_session'] = dt_for_first_bar_in_next_session
        path_dep_df.loc[dt, 'dt_when_trade_exited'] = dt_when_trade_exited

    trades_which_hit_limit_prices = (
        ~(path_dep_df.dt_when_limit_price_was_hit.eq(pd.NaT)) & (
            path_dep_df.dt_when_zone_was_hit.eq(pd.NaT) | (
                path_dep_df.dt_when_limit_price_was_hit.le(path_dep_df.dt_when_zone_was_hit)
            )
        )
    )
    path_dep_df_after_limit_prices = path_dep_df.loc[trades_which_hit_limit_prices].copy()

    previous_row = path_dep_df_after_limit_prices.iloc[0]
    it = iter(path_dep_df_after_limit_prices.iloc[1:].iterrows())
    for dt, row in it:
        if dt <= previous_row.dt_when_trade_exited or (
            (row.signals == previous_row.signals) and (dt < previous_row.dt_when_zone_changed)
        ):
            path_dep_df_after_limit_prices.drop(index=dt, inplace=True)

        else:
            previous_row = row.copy()

    next_session_if_exited_trade_at_eod = path_dep_df_after_limit_prices.loc[
        path_dep_df_after_limit_prices.dt_when_trade_exited == path_dep_df_after_limit_prices.dt_when_this_session_ended,
        'dt_for_first_bar_in_next_session'
    ]

    forecasts = pd.Series(0).reindex_like(price_index_series)
    forecasts[path_dep_df_after_limit_prices.dt_when_limit_price_was_hit] = signals[path_dep_df_after_limit_prices.index]
    forecasts[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    forecasts[next_session_if_exited_trade_at_eod] = 0
    forecasts = forecasts.ffill().fillna(0)

    new_long_limit_prices = pd.Series(np.nan).reindex_like(price_index_series)
    new_long_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    new_long_limit_prices[
        price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited[path_dep_df_after_limit_prices.signals > 0])
    ] = 0
    new_long_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_long_limit_prices = new_long_limit_prices.ffill().replace(0, np.nan)

    new_short_limit_prices = pd.Series(np.nan).reindex_like(price_index_series)
    new_short_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    new_short_limit_prices[
        price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited[path_dep_df_after_limit_prices.signals < 0])
    ] = 0
    new_short_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_short_limit_prices = new_short_limit_prices.ffill().replace(0, np.nan)

    stop_loss_levels = pd.Series(np.nan).reindex_like(price_index_series)
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    stop_loss_levels[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    stop_loss_levels[next_session_if_exited_trade_at_eod] = 0
    stop_loss_levels = stop_loss_levels.ffill().replace(0, np.nan)

    profit_target_levels = pd.Series(np.nan).reindex_like(price_index_series)
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    profit_target_levels[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)] = 0
    profit_target_levels[next_session_if_exited_trade_at_eod] = 0
    profit_target_levels = profit_target_levels.ffill().replace(0, np.nan)

    return dict(
        forecasts=forecasts,
        stop_loss_levels_after_slpt=stop_loss_levels,
        profit_target_levels_after_slpt=profit_target_levels,
        long_limit_prices_after_slpt=new_long_limit_prices,
        short_limit_prices_after_slpt=new_short_limit_prices,
        path_dep_df=path_dep_df_after_limit_prices,
    )

"""
"""
def get_signals_after_limit_price_is_hit(
    prices: pd.DataFrame,
    long_limit_prices: pd.Series,
    short_limit_prices: pd.Series,
    signals: pd.Series,
    long_zones: pd.DataFrame,
    short_zones: pd.DataFrame,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
):
    new_signals = signals.copy()
    new_long_limit_prices = long_limit_prices.copy()
    new_short_limit_prices = short_limit_prices.copy()
    new_long_stop_loss_levels = long_stop_loss_levels.copy()
    new_short_stop_loss_levels = short_stop_loss_levels.copy()
    new_long_profit_target_levels = long_profit_target_levels.copy()
    new_short_profit_target_levels = short_profit_target_levels.copy()

    # for each date in long/short signals, get limit price on that date.
    # See if prices hit limit price before a new zone is hit
    limit_price = np.nan
    price_index_series = prices.index.to_series()
    previous_signal = new_signals.iloc[0]
    dt_when_signal_or_zone_changes = new_signals.index[0]
    it = iter(new_signals.iloc[1:-1].items())
    for dt, signal in it:
        if signal != 0 and limit_price is np.nan:
            if signal == previous_signal and dt_when_signal_or_zone_changes >= dt:
                limit_price = np.nan
                new_signals[dt:dt_when_signal_or_zone_changes] = 0

                new_long_limit_prices[dt:dt_when_signal_or_zone_changes] = np.nan
                new_long_stop_loss_levels[dt:dt_when_signal_or_zone_changes] = np.nan
                new_long_profit_target_levels[dt:dt_when_signal_or_zone_changes] = np.nan

                new_short_limit_prices[dt:dt_when_signal_or_zone_changes] = np.nan
                new_short_stop_loss_levels[dt:dt_when_signal_or_zone_changes] = np.nan
                new_short_profit_target_levels[dt:dt_when_signal_or_zone_changes] = np.nan

                for _ in prices.loc[dt:dt_when_signal_or_zone_changes].index.to_series().iloc[:-2]:
                    try:
                        next(it)
                    except StopIteration:
                        break

                previous_signal = 0
                continue

            datetime_starting_from_next_bar = price_index_series.mask(
                price_index_series <= dt, np.nan
            ).dropna()
            if signal > 0:
                limit_price = new_long_limit_prices.loc[dt]
                try:
                    dt_when_limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'LOW'].le(limit_price).idxmax()
                except ValueError:
                    dt_when_limit_price_was_hit = pd.NaT
            else:
                limit_price = new_short_limit_prices.loc[dt]
                try:
                    dt_when_limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(limit_price).idxmax()
                except ValueError:
                    dt_when_limit_price_was_hit = pd.NaT

            if pd.isna(dt_when_limit_price_was_hit):    # Limit price was never hit
                signals_starting_from_this_bar = new_signals.mask(price_index_series < dt, np.nan).dropna()
                dt_when_signal_changes = signals_starting_from_this_bar.diff().iloc[1:].ne(0).idxmax()
                limit_price = np.nan
                new_signals[dt:dt_when_signal_changes] = 0
                if signal > 0:
                    new_long_limit_prices[dt:dt_when_signal_changes] = np.nan
                    new_long_stop_loss_levels[dt:dt_when_signal_changes] = np.nan
                    new_long_profit_target_levels[dt:dt_when_signal_changes] = np.nan

                else:
                    new_short_limit_prices[dt:dt_when_signal_changes] = np.nan
                    new_short_stop_loss_levels[dt:dt_when_signal_changes] = np.nan
                    new_short_profit_target_levels[dt:dt_when_signal_changes] = np.nan

                for _ in prices.loc[dt:dt_when_signal_changes].index.to_series().iloc[:-1]:
                    try:
                        next(it)
                    except StopIteration:
                        break

                previous_signal = 0
                continue

            if signal > 0:      # Ignore trade if it reaches opposing zone
                zone_to_be_hit = short_zones.loc[datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].cummax()
                dt_when_zone_was_hit = prices.loc[
                    datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].le(
                    zone_to_be_hit
                ).idxmax()
            else:
                zone_to_be_hit = long_zones.loc[datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].cummin()
                dt_when_zone_was_hit = prices.loc[
                    datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].ge(
                    zone_to_be_hit
                ).idxmax()

            if dt_when_zone_was_hit < dt_when_limit_price_was_hit:  # Did not enter trade
                limit_price = np.nan
                new_signals[dt:dt_when_zone_was_hit] = 0

                new_long_limit_prices[dt:dt_when_zone_was_hit] = np.nan
                new_long_stop_loss_levels[dt:dt_when_zone_was_hit] = np.nan
                new_long_profit_target_levels[dt:dt_when_zone_was_hit] = np.nan

                new_short_limit_prices[dt:dt_when_zone_was_hit] = np.nan
                new_short_stop_loss_levels[dt:dt_when_zone_was_hit] = np.nan
                new_short_profit_target_levels[dt:dt_when_zone_was_hit] = np.nan

                for _ in prices.loc[dt:dt_when_zone_was_hit].index.to_series().iloc[:-1]:
                    try:
                        next(it)
                    except StopIteration:
                        break

                previous_signal = 0
                continue

            else:   # Entered trade at dt_when_limit_price_was_hit
                new_signals[dt:dt_when_limit_price_was_hit] = 0
                new_signals[dt_when_limit_price_was_hit] = signal

                new_long_limit_prices[dt:dt_when_limit_price_was_hit] = np.nan
                new_short_limit_prices[dt:dt_when_limit_price_was_hit] = np.nan

                aux = new_long_stop_loss_levels[dt].copy()
                new_long_stop_loss_levels[dt:dt_when_limit_price_was_hit] = np.nan
                new_long_stop_loss_levels[dt_when_limit_price_was_hit] = aux

                aux = new_long_profit_target_levels[dt].copy()
                new_long_profit_target_levels[dt:dt_when_limit_price_was_hit] = np.nan
                new_long_profit_target_levels[dt_when_limit_price_was_hit] = aux

                aux = new_short_stop_loss_levels[dt].copy()
                new_short_stop_loss_levels[dt:dt_when_limit_price_was_hit] = np.nan
                new_short_stop_loss_levels[dt_when_limit_price_was_hit] = aux

                aux = new_short_profit_target_levels[dt].copy()
                new_short_profit_target_levels[dt:dt_when_limit_price_was_hit] = np.nan
                new_short_profit_target_levels[dt_when_limit_price_was_hit] = aux

                if signal > 0:
                    new_long_limit_prices[dt_when_limit_price_was_hit] = limit_price
                    new_short_limit_prices[dt_when_limit_price_was_hit] = np.nan
                else:
                    new_short_limit_prices[dt_when_limit_price_was_hit] = limit_price
                    new_long_limit_prices[dt_when_limit_price_was_hit] = np.nan

                zones_starting_from_this_bar = (
                    long_zones.mask(price_index_series < dt, np.nan).dropna() if signal > 0 else (
                        short_zones.mask(price_index_series < dt, np.nan).dropna()
                    )
                )
                dt_when_zone_changes = zones_starting_from_this_bar.diff().iloc[1:].ne(0).any(axis=1).idxmax()

                signals_starting_from_this_bar = new_signals.mask(price_index_series < dt, np.nan).dropna()
                dt_when_signal_changes = signals_starting_from_this_bar.diff().iloc[1:].ne(0).idxmax()

                dt_when_signal_or_zone_changes = min(dt_when_signal_changes, dt_when_zone_changes)

                previous_signal = signal
                for _ in prices.loc[dt:dt_when_limit_price_was_hit].index.to_series():
                    try:
                        next(it)
                    except StopIteration:
                        break
                next(it)

            limit_price = np.nan

        previous_signal = signal

    return dict(
        signals=new_signals,
        new_long_limit_prices=new_long_limit_prices,
        new_short_limit_prices=new_short_limit_prices,
        new_long_stop_loss_levels=new_long_stop_loss_levels,
        new_short_stop_loss_levels=new_short_stop_loss_levels,
        new_long_profit_target_levels=new_long_profit_target_levels,
        new_short_profit_target_levels=new_short_profit_target_levels,
    )


def apply_stop_loss_and_profit_target_to_signals(
    prices: pd.DataFrame,
    sessions: Session,
    signals: pd.Series,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
    long_limit_prices: pd.Series,
    short_limit_prices: pd.Series,
) -> dict:

    price_index_series = prices.index.to_series()
    trades = signals.diff().iloc[1:-1]
    entries = trades.loc[trades.ne(0) & signals.iloc[1:-1].ne(0)]
    datetime_when_price_crossed_sl_or_pt_for_trade = pd.Series(pd.NaT).reindex_like(entries)
    datetime_for_entries = datetime_when_price_crossed_sl_or_pt_for_trade.index.to_series()
    session_end_times_for_entries = pd.Series(
        [
            price_index_series.asof(
                pd.Timestamp(f'{x.date()} {sessions.end_time}').tz_convert(tz=datetime_for_entries.index.tz)
            ) for x in datetime_for_entries.index
        ],
        index=datetime_for_entries.index,
    )

    for dt, signal in entries.items():
        datetime_starting_from_next_bar = price_index_series.mask(
            price_index_series <= dt, pd.NaT
        ).dropna()
        datetime_when_price_crossed_sl_or_pt_for_trade.loc[dt] = (
            (prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(long_profit_target_levels.loc[dt]) | (
                prices.loc[datetime_starting_from_next_bar, 'LOW'].le(long_stop_loss_levels.loc[dt])
            )).idxmax() if signal > 0 else (
                (prices.loc[datetime_starting_from_next_bar, 'LOW'].le(short_profit_target_levels.loc[dt]) | (
                    prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(short_stop_loss_levels.loc[dt])
                )).idxmax()
            )
        )
        datetime_when_price_crossed_sl_or_pt_for_trade[dt] = min(   # EOD exit
            datetime_when_price_crossed_sl_or_pt_for_trade[dt],
            session_end_times_for_entries[dt]
        )

    # at a certain date, get first trade (entry) where you would be in the market
    # get, for a certain signal, whether the prices hit stop loss or profit target on the previous signal's entry
    datetime_when_price_crossed_sl_or_pt_for_trade.dropna(inplace=True)
    previous_exit_dt = datetime_when_price_crossed_sl_or_pt_for_trade.iloc[0]
    it = iter(datetime_when_price_crossed_sl_or_pt_for_trade.iloc[1:].items())
    for trade_dt, exit_dt in it:
        if trade_dt <= previous_exit_dt:
            datetime_when_price_crossed_sl_or_pt_for_trade.drop(index=trade_dt, inplace=True)
        else:
            previous_exit_dt = exit_dt

    no_entries_at_these_dates = [x for x in signals.index if x not in datetime_when_price_crossed_sl_or_pt_for_trade.index]

    signals.loc[no_entries_at_these_dates] = np.nan
    signals.loc[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    signals = signals.ffill().fillna(0)

    long_limit_prices.loc[no_entries_at_these_dates] = np.nan
    long_limit_prices[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    long_limit_prices = long_limit_prices.ffill().replace(0, np.nan)

    short_limit_prices.loc[no_entries_at_these_dates] = np.nan
    short_limit_prices[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    short_limit_prices = short_limit_prices.ffill().replace(0, np.nan)

    long_stop_loss_levels.loc[no_entries_at_these_dates] = np.nan
    long_stop_loss_levels[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    long_stop_loss_levels = long_stop_loss_levels.ffill().replace(0, np.nan)

    short_stop_loss_levels.loc[no_entries_at_these_dates] = np.nan
    short_stop_loss_levels[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    short_stop_loss_levels = short_stop_loss_levels.ffill().replace(0, np.nan)

    long_profit_target_levels.loc[no_entries_at_these_dates] = np.nan
    long_profit_target_levels[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    long_profit_target_levels = long_profit_target_levels.ffill().replace(0, np.nan)

    short_profit_target_levels.loc[no_entries_at_these_dates] = np.nan
    short_profit_target_levels[datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    short_profit_target_levels = short_profit_target_levels.ffill().replace(0, np.nan)

    stop_loss_levels = long_stop_loss_levels.add(short_stop_loss_levels, fill_value=0).replace(0, np.nan)
    profit_target_levels = long_profit_target_levels.add(short_profit_target_levels, fill_value=0).replace(0, np.nan)

    return dict(
        forecasts=signals,
        stop_loss_levels_after_slpt=stop_loss_levels,
        profit_target_levels_after_slpt=profit_target_levels,
        long_limit_prices_after_slpt=long_limit_prices,
        short_limit_prices_after_slpt=short_limit_prices,
    )

"""
if __name__ == "__main__":
    import doctest

    doctest.testmod()

    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf

    from private.systems.orion.rules.orion import orion

    from sysdata.sim.db_futures_sim_data import dbFuturesSimData

    data = dbFuturesSimData()

    price_bars = data.get_backadjusted_futures_price('CL')
    price_bars = price_bars.loc[price_bars['FINAL'] != 0.0]

    sessions = data.get_sessions_for_instrument('CL')

    small_price_bars = price_bars.resample('5T').agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    small_price_bars = apply_sessions_to_aggregated_data(small_price_bars, sessions)

    big_price_bars = price_bars.resample('30T').agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    big_price_bars = apply_sessions_to_aggregated_data(big_price_bars, sessions)

    orion_trades = orion(price_bars, sessions, rr=2.5)
    long_signals = orion_trades['long_signals']
    short_signals = orion_trades['short_signals']

    new_orion_trades = apply_limit_prices_slpt_to_signals(
        prices=small_price_bars,
        long_signals=long_signals,
        short_signals=short_signals,
        sessions=sessions,
        long_limit_prices=orion_trades['long_limit_prices'],
        short_limit_prices=orion_trades['short_limit_prices'],
        long_stop_loss_levels=orion_trades['long_stop_loss_prices'],
        short_stop_loss_levels=orion_trades['short_stop_loss_prices'],
        long_profit_target_levels=orion_trades['long_profit_taker'],
        short_profit_target_levels=orion_trades['short_profit_taker'],
    )
    new_signals = new_orion_trades['forecasts']

    where_values = new_orion_trades['long_limit_prices_after_slpt'].add(new_orion_trades['short_limit_prices_after_slpt'], fill_value=0)
    new_apds = [
        mpf.make_addplot(small_price_bars['LOW'].where(new_signals > 0, np.nan), type='scatter', marker='^'),
        mpf.make_addplot(small_price_bars['HIGH'].where(new_signals < 0, np.nan), type='scatter', marker='v'),
        mpf.make_addplot(orion_trades['red_fractal_prices'], type='scatter', color='red', marker='v'),
        mpf.make_addplot(orion_trades['green_fractal_prices'], type='scatter', color='green', marker='^'),
        mpf.make_addplot(new_orion_trades['long_limit_prices_after_slpt'], type='line', color='blue'),
        mpf.make_addplot(new_orion_trades['short_limit_prices_after_slpt'], type='line', color='blue'),
        mpf.make_addplot(
            new_orion_trades['stop_loss_levels_after_slpt'], type='line', color='maroon',
            fill_between=dict(
                y1=new_orion_trades['stop_loss_levels_after_slpt'].values,
                y2=where_values.values,
                where=~(where_values.isna()).values,
                alpha=0.5,
                color='red'
            )
        ),
        mpf.make_addplot(
            new_orion_trades['profit_target_levels_after_slpt'], type='line', color='green',
            fill_between=dict(
                y1=new_orion_trades['profit_target_levels_after_slpt'].values,
                y2=where_values.values,
                where=~(where_values.isna()).values,
                alpha=0.5,
                color='green'
            )
        ),
    ]

    when_price_hit_which_demand_zone = orion_trades['when_price_hit_which_demand_zone'].reset_index()[0]
    when_price_hit_which_supply_zone = orion_trades['when_price_hit_which_supply_zone'].reset_index()[0]

    demand_zones_dts = pd.Series(
        [x[i] for x in when_price_hit_which_demand_zone.values for i in range(len(x))]).drop_duplicates()
    for dt, demand_zone in orion_trades['demand_zones'].iterrows():

        dt_when_to_cancel_zone = small_price_bars.loc[dt:, 'FINAL'].lt(demand_zone.LOW)
        dt_when_to_cancel_zone = small_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        demand_zone = pd.DataFrame(dict(zip(demand_zone.index, demand_zone.values)),
                                   index=small_price_bars.loc[dt:dt_when_to_cancel_zone].index).reindex_like(
            small_price_bars[['HIGH', 'LOW']]
        )

        if len(demand_zone) > 0:
            new_apds.append(
                mpf.make_addplot(
                    demand_zone.LOW, type='line', color='blue',
                    fill_between=dict(
                        y1=demand_zone.LOW.values,
                        y2=demand_zone.HIGH.values,
                        where=~(demand_zone.HIGH.isna()).values,
                        alpha=0.5,
                        color='blue'
                    )
                )
            )

    supply_zones_dts = pd.Series(
        [x[i] for x in when_price_hit_which_supply_zone.values for i in range(len(x))]).drop_duplicates()
    for dt, supply_zone in orion_trades['supply_zones'].iterrows():
        dt_when_to_cancel_zone = small_price_bars.loc[dt:, 'FINAL'].gt(supply_zone.HIGH)
        dt_when_to_cancel_zone = small_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        supply_zone = pd.DataFrame(dict(zip(supply_zone.index, supply_zone.values)),
                                   index=small_price_bars.loc[dt:dt_when_to_cancel_zone].index).reindex_like(
            small_price_bars[['HIGH', 'LOW']]
        )

        if len(supply_zone) > 0:
            new_apds.append(
                mpf.make_addplot(
                    supply_zone.HIGH, type='line', color='orange',
                    fill_between=dict(
                        y1=supply_zone.HIGH.values,
                        y2=supply_zone.LOW.values,
                        where=~(supply_zone.HIGH.isna()).values,
                        alpha=0.5,
                        color='orange'
                    )
                )
            )

    mpf.plot(
        small_price_bars[['OPEN', 'HIGH', 'LOW', 'FINAL']].rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
        type='candle',
        show_nontrading=False,
        addplot=new_apds,
    )

    orion_trades_df = pd.DataFrame(orion_trades)
    # orion_trades_which_hit_limit_prices_df = pd.DataFrame(orion_trades_which_hit_limit_prices)
    new_orion_trades_df = pd.DataFrame({k: v for k, v in new_orion_trades.items() if k != 'path_dep_df'})

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
            previous_row = copy(row)
        elif dt <= previous_row.dt_when_trade_exited:
            print(f'*** Signal generated at {dt} before previous order exited at {previous_row.dt_when_trade_exited}. Dropping signal ***')
            path_dep_df_after_limit_prices.drop(index=dt, inplace=True)
        else:
            print(f'*** Everything seems fine over here. Keeping signal ***')
            previous_dt = dt
            previous_row = copy(row)

    print(f'******* Done checking signals. There were {len(path_dep_df_after_limit_prices.index)} trades ' + (
        f'from {price_index_series.iloc[0]} to {price_index_series.iloc[-1]} *******')
    )
    next_session_if_exited_trade_at_eod = path_dep_df_after_limit_prices.loc[
        path_dep_df_after_limit_prices.dt_when_trade_exited == path_dep_df_after_limit_prices.dt_when_this_session_ended,
        'dt_for_first_bar_in_next_session'
    ].dropna()
    first_index_after_exit = price_index_series.shift(-1).loc[price_index_series.asof(path_dep_df_after_limit_prices.dt_when_trade_exited)]
    first_index_after_exit_for_long_trades = price_index_series.shift(-1).loc[price_index_series.asof(
        path_dep_df_after_limit_prices.dt_when_trade_exited.loc[path_dep_df_after_limit_prices.signals > 0]
    )]
    first_index_after_exit_for_short_trades = price_index_series.shift(-1).loc[price_index_series.asof(
        path_dep_df_after_limit_prices.dt_when_trade_exited.loc[path_dep_df_after_limit_prices.signals < 0]
    )]

    if price_index_series.iloc[-1] in first_index_after_exit.index:
        first_index_after_exit.drop(index=price_index_series.iloc[-1], inplace=True)
    if price_index_series.iloc[-1] in first_index_after_exit_for_long_trades.index:
        first_index_after_exit_for_long_trades.drop(index=price_index_series.iloc[-1], inplace=True)
    if price_index_series.iloc[-1] in first_index_after_exit_for_short_trades.index:
        first_index_after_exit_for_short_trades.drop(index=price_index_series.iloc[-1], inplace=True)

    forecasts = pd.Series(np.nan, index=prices.index)
    forecasts[path_dep_df_after_limit_prices.dt_when_limit_price_was_hit] = path_dep_df_after_limit_prices.signals
    forecasts[first_index_after_exit] = 0
    forecasts[next_session_if_exited_trade_at_eod] = 0
    forecasts = forecasts.ffill().fillna(0)

    new_long_limit_prices = pd.Series(np.nan, index=prices.index)
    new_long_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    new_long_limit_prices[first_index_after_exit_for_long_trades] = 0
    new_long_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_long_limit_prices = new_long_limit_prices.ffill().replace(0, np.nan)

    new_short_limit_prices = pd.Series(np.nan, index=prices.index)
    new_short_limit_prices[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_limit_prices.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    new_short_limit_prices[first_index_after_exit_for_short_trades] = 0
    new_short_limit_prices[next_session_if_exited_trade_at_eod] = 0
    new_short_limit_prices = new_short_limit_prices.ffill().replace(0, np.nan)

    stop_loss_levels = pd.Series(np.nan, index=prices.index)
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    stop_loss_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_stop_loss_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    stop_loss_levels[first_index_after_exit] = 0
    stop_loss_levels[next_session_if_exited_trade_at_eod] = 0
    stop_loss_levels = stop_loss_levels.ffill().replace(0, np.nan)

    profit_target_levels = pd.Series(np.nan, index=prices.index)
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals > 0]
    ] = long_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals > 0].index]
    profit_target_levels[
        path_dep_df_after_limit_prices.dt_when_limit_price_was_hit[path_dep_df_after_limit_prices.signals < 0]
    ] = short_profit_target_levels.loc[path_dep_df_after_limit_prices.loc[path_dep_df_after_limit_prices.signals < 0].index]
    profit_target_levels[first_index_after_exit] = 0
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf

    from private.systems.orion.rules.orion import orion

    from sysdata.sim.db_futures_sim_data import dbFuturesSimData

    data = dbFuturesSimData()

    instrument_code = 'CL'
    rr = 2.0
    small_timeframe = '5min'
    big_timeframe = '30min'

    price_bars = data.get_backadjusted_futures_price(instrument_code)
    price_bars = price_bars.loc[price_bars['FINAL'] != 0.0]
    price_bars = price_bars.iloc[1:]

    # price_bars = price_bars.loc['2024-01-02 00:00:00':]
    # price_bars = pd.read_csv(get_resolved_pathname('data.NYMEX_DL_CL1!, 1') + '.csv', index_col=[0], parse_dates=True)[['open', 'high', 'low', 'close']].rename(
    #     columns=dict(open='OPEN', high='HIGH', low='LOW', close='FINAL')
    # )
    # price_bars['VOLUME'] = 0
    # price_bars = price_bars.loc[price_bars.index.to_series().asof('2024-05-30 17:00:00-05:00'):price_bars.index.to_series().asof('2024-06-30 16:00:00-05:00')]

    sessions = data.get_sessions_for_instrument(instrument_code)

    # small_price_bars = price_bars.resample('5T').agg(
    #     {
    #         'OPEN': 'first',
    #         'HIGH': 'max',
    #         'LOW': 'min',
    #         'FINAL': 'last',
    #         'VOLUME': 'sum',
    #     }
    # )
    # small_price_bars = apply_sessions_to_aggregated_data(small_price_bars, sessions)
    #
    # big_price_bars = price_bars.resample('30T').agg(
    #     {
    #         'OPEN': 'first',
    #         'HIGH': 'max',
    #         'LOW': 'min',
    #         'FINAL': 'last',
    #         'VOLUME': 'sum',
    #     }
    # )
    # big_price_bars = apply_sessions_to_aggregated_data(big_price_bars, sessions)

    orion_trades = orion(price_bars, sessions, small_timeframe=small_timeframe, big_timeframe=big_timeframe, rr=rr)

    small_price_bars = orion_trades['small_price_bars']
    big_price_bars = orion_trades['big_price_bars']
    long_signals = orion_trades['long_signals']
    short_signals = orion_trades['short_signals']

    swing_highs = orion_trades['swing_highs']
    swing_highs.loc[swing_highs.eq(0)] = np.nan
    swing_highs = swing_highs.reindex_like(small_price_bars['FINAL'])

    swing_lows = orion_trades['swing_lows']
    swing_lows.loc[swing_lows.eq(0)] = np.nan
    swing_lows = swing_lows.reindex_like(small_price_bars['FINAL'])

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
        mpf.make_addplot(swing_highs, type='scatter', color='purple', marker='x'),
        mpf.make_addplot(swing_lows, type='scatter', color='yellow', marker='x'),
        mpf.make_addplot(orion_trades['red_fractals_prices'], type='scatter', color='red', marker='v'),
        mpf.make_addplot(orion_trades['green_fractals_prices'], type='scatter', color='green', marker='^'),
        mpf.make_addplot(new_orion_trades['long_limit_prices_after_slpt'], type='line', color='blue'),
        mpf.make_addplot(new_orion_trades['short_limit_prices_after_slpt'], type='line', color='blue'),
        # mpf.make_addplot(small_price_bars.loc[orion_trades['demand_zones'].index, 'HIGH'].reindex_like(small_price_bars['FINAL']), type='scatter', color='pink',
        #                  marker='o'),
        # mpf.make_addplot(small_price_bars.loc[orion_trades['supply_zones'].index, 'LOW'].reindex_like(small_price_bars['FINAL']), type='scatter', color='pink',
        #                  marker='o'),
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
                                   index=small_price_bars.loc[demand_zone.idx:dt_when_to_cancel_zone].index).reindex_like(
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
                                   index=small_price_bars.loc[supply_zone.idx:dt_when_to_cancel_zone].index).reindex_like(
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

    path_dep_df = new_orion_trades.pop('path_dep_df')
    new_orion_trades = pd.DataFrame(new_orion_trades)

    price_index_series = small_price_bars.index.to_series()
    trades = pd.DataFrame(dict(signal=0, position=0.0, entry_price=np.nan, exit_price=np.nan),
                          index=small_price_bars.index)
    trades_summary = pd.DataFrame(
        dict(
            signal=np.nan, signal_dt=pd.NaT, position=np.nan,
            entry_price=np.nan, entry_dt=pd.NaT, exit_price=np.nan, exit_dt=pd.NaT,
            exit_method='',
        ),
        index=path_dep_df.index
    )
    multiplier = 1000.0
    risk_per_trade_pct_capital = 0.05
    capital_allocated_to_instrument = 1000000
    for signal_dt, trade_metadata in path_dep_df.iterrows():

        entry_dt = trade_metadata.dt_when_limit_price_was_hit
        exit_dt = price_index_series.asof(trade_metadata.dt_when_trade_exited)
        idxs_in_the_market = trades.loc[entry_dt:exit_dt].index
        signal = trade_metadata.signals

        trades.loc[idxs_in_the_market, 'signal'] = signal
        trades.loc[idxs_in_the_market, 'entry_price'] = small_price_bars.shift(1).loc[idxs_in_the_market, 'FINAL']
        trades.loc[idxs_in_the_market, 'exit_price'] = small_price_bars.loc[idxs_in_the_market, 'FINAL']

        if signal > 0:
            limit_price = "long_limit_prices_after_slpt"
        else:
            limit_price = "short_limit_prices_after_slpt"

        trades.loc[entry_dt, 'entry_price'] = new_orion_trades.loc[entry_dt, limit_price]
        trade_risk = (
            multiplier * (new_orion_trades.loc[entry_dt, limit_price] - new_orion_trades.loc[
                entry_dt, 'stop_loss_levels_after_slpt'
            ])
        )
        position_size_in_contracts = round(
            (capital_allocated_to_instrument * risk_per_trade_pct_capital) / trade_risk
        ) if trade_risk != 0 else 0

        trades.loc[idxs_in_the_market, 'position'] = position_size_in_contracts

        if trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_this_session_ended:
            exit_method = "EOD"
            trades.loc[exit_dt, 'exit_price'] = small_price_bars.FINAL.asof(trade_metadata.dt_when_trade_exited)
        elif trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_stop_loss_was_hit:
            exit_method = "Stop Loss"
            trades.loc[exit_dt, 'exit_price'] = new_orion_trades.stop_loss_levels_after_slpt.asof(exit_dt)
        elif trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_profit_target_was_hit:
            exit_method = "Profit Target"
            trades.loc[exit_dt, 'exit_price'] = new_orion_trades.profit_target_levels_after_slpt.asof(exit_dt)
        else:
            exit_method = "None"
            input("WARNING! Continue?")

        trades_summary.loc[signal_dt] = (
            signal, signal_dt, position_size_in_contracts,
            trades.loc[entry_dt, 'entry_price'], entry_dt,
            trades.loc[exit_dt, 'exit_price'], exit_dt,
            exit_method
        )

    profit = trades['position'] * (trades['exit_price'] - trades['entry_price'])
    returns = profit / trades['entry_price']
    returns_relative_to_capital = profit / capital_allocated_to_instrument

    profit.fillna(0, inplace=True)
    returns.fillna(0, inplace=True)
    returns_relative_to_capital.fillna(0, inplace=True)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(profit.cumsum())
    plt.title(f'{instrument_code} {rr}/1')

    trades_summary['signal'] = ["L" if x == 1 else "S" for x in trades_summary['signal']]
    trades_summary.reset_index(inplace=True, drop=True)
    trades_summary.to_csv(get_resolved_pathname('private.systems.orion.trades_summary') + f'_{instrument_code}_{rr}-to-1.csv', sep='\t')

    ######################################
    big_apds = [
        mpf.make_addplot(swing_highs.reindex_like(big_price_bars['FINAL']), type='scatter', color='purple', marker='x'),
        mpf.make_addplot(swing_lows.reindex_like(big_price_bars['FINAL']), type='scatter', color='yellow', marker='x'),
        mpf.make_addplot(big_price_bars.loc[orion_trades['demand_zones'].index, 'HIGH'].reindex_like(big_price_bars['FINAL']), type='scatter', color='pink',
                         marker='o'),
        mpf.make_addplot(big_price_bars.loc[orion_trades['supply_zones'].index, 'LOW'].reindex_like(big_price_bars['FINAL']), type='scatter', color='pink',
                         marker='o'),
    ]

    for dt, demand_zone in orion_trades['demand_zones'].iterrows():

        dt_when_to_cancel_zone = big_price_bars.loc[dt:, 'FINAL'].lt(demand_zone.LOW)
        dt_when_to_cancel_zone = big_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        demand_zone = pd.DataFrame(dict(zip(demand_zone.index, demand_zone.values)),
                                   index=big_price_bars.loc[
                                         demand_zone.idx:dt_when_to_cancel_zone].index).reindex_like(
            big_price_bars[['HIGH', 'LOW']]
        )

        if len(demand_zone) > 0:
            big_apds.append(
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

    for dt, supply_zone in orion_trades['supply_zones'].iterrows():
        dt_when_to_cancel_zone = big_price_bars.loc[dt:, 'FINAL'].gt(supply_zone.HIGH)
        dt_when_to_cancel_zone = big_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        supply_zone = pd.DataFrame(dict(zip(supply_zone.index, supply_zone.values)),
                                   index=big_price_bars.loc[
                                         supply_zone.idx:dt_when_to_cancel_zone].index).reindex_like(
            big_price_bars[['HIGH', 'LOW']]
        )

        if len(supply_zone) > 0:
            big_apds.append(
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
        big_price_bars[['OPEN', 'HIGH', 'LOW', 'FINAL']].rename(
            columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
        type='candle',
        show_nontrading=False,
        addplot=big_apds,
    )

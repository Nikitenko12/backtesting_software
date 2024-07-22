import pandas as pd
import numpy as np

from sysobjects.sessions import Session
from systems.stage import SystemStage
from systems.system_cache import input, diagnostic, output


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

        long_limit_prices = signals_dict['long_limit_prices']
        short_limit_prices = signals_dict['short_limit_prices']
        signals = signals_dict['signals']
        long_zones = signals_dict['long_zones']
        short_zones = signals_dict['short_zones']
        long_stop_loss_levels = signals_dict['long_stop_loss_prices']
        short_stop_loss_levels = signals_dict['short_stop_loss_prices']
        long_profit_target_levels = signals_dict['long_profit_taker']
        short_profit_target_levels = signals_dict['short_profit_taker']

        return (
            prices,
            sessions,
            long_limit_prices,
            short_limit_prices,
            signals,
            long_zones,
            short_zones,
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
            long_limit_prices,
            short_limit_prices,
            signals,
            long_zones,
            short_zones,
            long_stop_loss_levels,
            short_stop_loss_levels,
            long_profit_target_levels,
            short_profit_target_levels,
        ) = self.get_inputs(instrument_code)

        signals_after_limit_prices = get_signals_after_limit_price_is_hit(
            prices=prices,
            long_limit_prices=long_limit_prices,
            short_limit_prices=short_limit_prices,
            signals=signals,
            long_zones=long_zones,
            short_zones=short_zones,
            long_stop_loss_levels=long_stop_loss_levels,
            short_stop_loss_levels=short_stop_loss_levels,
            long_profit_target_levels=long_profit_target_levels,
            short_profit_target_levels=short_profit_target_levels,
        )
        updated_signals_dict = apply_stop_loss_and_profit_target_to_signals(
            prices=prices,
            sessions=sessions,
            signals=signals_after_limit_prices['signals'],
            long_stop_loss_levels=signals_after_limit_prices['new_long_stop_loss_levels'],
            short_stop_loss_levels=signals_after_limit_prices['new_short_stop_loss_levels'],
            long_profit_target_levels=signals_after_limit_prices['new_long_profit_target_levels'],
            short_profit_target_levels=signals_after_limit_prices['new_short_profit_target_levels'],
            long_limit_prices=signals_after_limit_prices['new_long_limit_prices'],
            short_limit_prices=signals_after_limit_prices['new_short_limit_prices'],
        )

        return updated_signals_dict

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


def apply_limit_prices_to_signals(
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
    limit_price = np.nan
    price_index_series = prices.index.to_series()
    trades = signals.loc[signals.ne(0)]
    it = iter(trades.iloc[:-1].items())

    limit_prices_df = pd.DataFrame(columns=['signal_dt', 'dt_when_limit_price_was_hit', 'dt_when_zone_was_hit'])

    for dt, signal in it:
        datetime_starting_from_next_bar = price_index_series.mask(
            price_index_series <= dt, np.nan
        ).dropna()
        if signal > 0:
            limit_price = long_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'LOW'].le(limit_price)
            dt_when_limit_price_was_hit = pd.NaT if limit_price_was_hit.eq(
                False).all() else limit_price_was_hit.idxmax()

            zone_to_be_hit = short_zones.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].cummax()
            zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].le(
                zone_to_be_hit
            )

            zones_starting_from_this_bar = long_zones.mask(price_index_series < dt, np.nan).dropna()
            zone_changes = zones_starting_from_this_bar.diff().iloc[1:].ne(0).any(axis=1)

        else:
            limit_price = short_limit_prices.loc[dt]
            limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(limit_price)

            zone_to_be_hit = long_zones.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].cummin()
            zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].ge(
                zone_to_be_hit
            )

            zones_starting_from_this_bar = short_zones.mask(price_index_series < dt, np.nan).dropna()


        zone_changes = zones_starting_from_this_bar.diff().iloc[1:].ne(0).any(axis=1)

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf

    from private.systems.orion.rules.orion import orion

    # big_price_bars = \
    # pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 30.csv', index_col=[0], parse_dates=True)[
    #     ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='FINAL'))
    # big_price_bars = big_price_bars.reindex(big_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))
    # small_price_bars = \
    # pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 5.csv', index_col=[0], parse_dates=True)[
    #     ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='FINAL'))
    # small_price_bars = small_price_bars.reindex(small_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))

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
    big_price_bars = price_bars.resample('30T').agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )

    datetime_big = big_price_bars.index.to_series()
    session_end_times_big = pd.Series([pd.Timestamp(f'{x.date()} {sessions.end_time}') for x in datetime_big],
                                      index=datetime_big.index)
    session_start_times_big = pd.Series([pd.Timestamp(f'{x.date()} {sessions.start_time}') for x in datetime_big],
                                        index=datetime_big.index)
    big_price_bars = big_price_bars.loc[
        ~(
                (
                    big_price_bars.index.to_series().ge(session_end_times_big)
                ) & (
                    big_price_bars.index.to_series().lt(session_start_times_big)
                )
        )
    ]

    datetime_small = small_price_bars.index.to_series()
    session_end_times_small = pd.Series([pd.Timestamp(f'{x.date()} {sessions.end_time}') for x in datetime_small],
                                  index=datetime_small.index)
    session_start_times_small = pd.Series([pd.Timestamp(f'{x.date()} {sessions.start_time}') for x in datetime_small],
                                    index=datetime_small.index)
    small_price_bars = small_price_bars.loc[
        ~(
                (
                    small_price_bars.index.to_series().ge(session_end_times_small)
                ) & (
                    small_price_bars.index.to_series().lt(session_start_times_small)
                )
        )
    ]

    orion_trades = orion(price_bars, sessions)
    signals = orion_trades['signals']

    # apds = [
    #     mpf.make_addplot(small_price_bars['LOW'].where(signals > 0, np.nan), type='scatter', marker='^'),
    #     mpf.make_addplot(small_price_bars['HIGH'].where(signals < 0, np.nan), type='scatter', marker='v'),
    #     mpf.make_addplot(orion_trades['long_stop_loss_prices'], type='line'),
    #     mpf.make_addplot(orion_trades['long_profit_taker'], type='line'),
    #     mpf.make_addplot(orion_trades['short_stop_loss_prices'], type='line'),
    #     mpf.make_addplot(orion_trades['short_profit_taker'], type='line'),
    # ]
    # mpf.plot(
    #     small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
    #     type='candle',
    #     show_nontrading=False,
    #     addplot=apds,
    # )

    orion_trades_which_hit_limit_prices = get_signals_after_limit_price_is_hit(
        prices=small_price_bars,
        signals=signals,
        long_limit_prices=orion_trades['long_limit_prices'],
        short_limit_prices=orion_trades['short_limit_prices'],
        long_zones=orion_trades['long_zones'],
        short_zones=orion_trades['short_zones'],
        long_stop_loss_levels=orion_trades['long_stop_loss_prices'],
        short_stop_loss_levels=orion_trades['short_stop_loss_prices'],
        long_profit_target_levels=orion_trades['long_profit_taker'],
        short_profit_target_levels=orion_trades['short_profit_taker'],
    )

    new_orion_trades = apply_stop_loss_and_profit_target_to_signals(
        prices=small_price_bars,
        sessions=sessions,
        signals=orion_trades_which_hit_limit_prices['signals'],
        long_stop_loss_levels=orion_trades_which_hit_limit_prices['new_long_stop_loss_levels'],
        short_stop_loss_levels=orion_trades_which_hit_limit_prices['new_short_stop_loss_levels'],
        long_profit_target_levels=orion_trades_which_hit_limit_prices['new_long_profit_target_levels'],
        short_profit_target_levels=orion_trades_which_hit_limit_prices['new_short_profit_target_levels'],
        long_limit_prices=orion_trades_which_hit_limit_prices['new_long_limit_prices'],
        short_limit_prices=orion_trades_which_hit_limit_prices['new_short_limit_prices'],
    )
    new_signals = new_orion_trades['forecasts']

    where_values = new_orion_trades['long_limit_prices_after_slpt'].add(new_orion_trades['short_limit_prices_after_slpt'], fill_value=0)
    new_apds = [
        mpf.make_addplot(small_price_bars['LOW'].where(new_signals > 0, np.nan), type='scatter', marker='^'),
        mpf.make_addplot(small_price_bars['HIGH'].where(new_signals < 0, np.nan), type='scatter', marker='v'),
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
    mpf.plot(
        small_price_bars[['OPEN', 'HIGH', 'LOW', 'FINAL']].rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
        type='candle',
        show_nontrading=False,
        addplot=new_apds,
    )

    orion_trades_df = pd.DataFrame({x: orion_trades[x] for x in list(orion_trades.keys())[:-2]})
    orion_trades_which_hit_limit_prices_df = pd.DataFrame(orion_trades_which_hit_limit_prices)
    new_orion_trades_df = pd.DataFrame(new_orion_trades)

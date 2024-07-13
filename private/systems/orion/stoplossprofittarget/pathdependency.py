import pandas as pd
import numpy as np

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
    new_signals = signals.copy() # shift(1)
    new_long_limit_prices = long_limit_prices.copy()
    new_short_limit_prices = short_limit_prices.copy()
    new_long_stop_loss_levels = long_stop_loss_levels.copy()
    new_short_stop_loss_levels = short_stop_loss_levels.copy()
    new_long_profit_target_levels = long_profit_target_levels.copy()
    new_short_profit_target_levels = short_profit_target_levels.copy()

    # for each date in long/short signals, get limit price on that date.
    # See if prices hit limit price before a new zone is hit.
    it = iter(list(new_signals.items())[:-1])
    for dt, signal in it:
        if signal == 0:
            continue
        datetime_starting_from_next_bar = prices.index.to_series()
        datetime_starting_from_next_bar = datetime_starting_from_next_bar.mask(
            datetime_starting_from_next_bar <= dt, np.nan
        ).dropna()
        if signal > 0:
            limit_price = new_long_limit_prices.loc[dt]
            try:
                dt_when_limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'LOW'].le(limit_price).idxmax()
            except ValueError:
                dt_when_limit_price_was_hit = pd.NA
        else:
            limit_price = new_short_limit_prices.loc[dt]
            try:
                dt_when_limit_price_was_hit = prices.loc[datetime_starting_from_next_bar, 'HIGH'].ge(limit_price).idxmax()
            except ValueError:
                dt_when_limit_price_was_hit = pd.NA

        if pd.isna(dt_when_limit_price_was_hit):   # Limit price was never hit
            continue

        if signal > 0:
            zone_to_be_hit = long_zones.loc[datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].cummax()
            dt_when_zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].le(   ## FIXME (dt_when_limit_price_was_hit + 1)??
                zone_to_be_hit
            ).idxmax()
        else:
            zone_to_be_hit = short_zones.loc[datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'LOW'].cummin()
            dt_when_zone_was_hit = prices.loc[
                datetime_starting_from_next_bar.loc[:dt_when_limit_price_was_hit], 'HIGH'].ge(
                zone_to_be_hit
            ).idxmax()

        if dt_when_zone_was_hit < dt_when_limit_price_was_hit:  # Did not enter trade
            new_signals[dt] = 0
            if signal > 0:
                new_long_limit_prices[dt] = np.nan
            else:
                new_short_limit_prices[dt] = np.nan
        else:   # Entered trade at dt_when_limit_price_was_hit
            new_signals[dt:dt_when_limit_price_was_hit] = 0
            new_signals[dt_when_limit_price_was_hit] = signal

            if signal > 0:
                new_long_limit_prices[dt:dt_when_limit_price_was_hit] = np.nan
                new_long_limit_prices[dt_when_limit_price_was_hit] = limit_price

                new_long_stop_loss_levels[dt_when_limit_price_was_hit] = new_long_stop_loss_levels[dt]
                new_long_stop_loss_levels[dt:dt_when_limit_price_was_hit] = np.nan

                new_long_profit_target_levels[dt_when_limit_price_was_hit] = new_long_profit_target_levels[dt]
                new_long_profit_target_levels[dt:dt_when_limit_price_was_hit] = np.nan
            else:
                new_short_limit_prices[dt:dt_when_limit_price_was_hit] = np.nan
                new_short_limit_prices[dt_when_limit_price_was_hit] = limit_price

                new_short_stop_loss_levels[dt_when_limit_price_was_hit] = new_short_stop_loss_levels[dt]
                new_short_stop_loss_levels[dt:dt_when_limit_price_was_hit] = np.nan

                new_short_profit_target_levels[dt_when_limit_price_was_hit] = new_short_profit_target_levels[dt]
                new_short_profit_target_levels[dt:dt_when_limit_price_was_hit] = np.nan

            for _ in prices.loc[dt:dt_when_limit_price_was_hit].index:
                next(it)

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
    signals: pd.Series,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
    long_limit_prices: pd.Series,
    short_limit_prices: pd.Series,
) -> dict:

    signals = signals.shift(1)

    # get index where prices crossed stop loss and profit target for each trade
    datetime_when_price_crossed_sl_or_pt_for_trade = pd.Series(None).reindex_like(signals.loc[signals != 0].iloc[1:])
    for dt, signal in (signals.loc[signals != 0].iloc[1:-1]).items():
        datetime_starting_from_next_bar = prices.index.to_series()
        datetime_starting_from_next_bar = datetime_starting_from_next_bar.mask(
            datetime_starting_from_next_bar <= dt, np.nan
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

    # at a certain date, get first trade (entry) where you would be in the market
    # get, for a certain signal, whether the prices hit stop loss or profit target on the previous signal's entry
    new_datetime_when_price_crossed_sl_or_pt_for_trade = datetime_when_price_crossed_sl_or_pt_for_trade.copy()
    for i in range(1, len(datetime_when_price_crossed_sl_or_pt_for_trade)-1):
        if new_datetime_when_price_crossed_sl_or_pt_for_trade.loc[
            datetime_when_price_crossed_sl_or_pt_for_trade.index[i]
        ] <= (
            new_datetime_when_price_crossed_sl_or_pt_for_trade.shift(1).loc[
                datetime_when_price_crossed_sl_or_pt_for_trade.index[i]
            ]
        ):
            new_datetime_when_price_crossed_sl_or_pt_for_trade.drop(
                datetime_when_price_crossed_sl_or_pt_for_trade.index[i], inplace=True
            )

    # Removing trades which entered before last trade exited
    new_datetime_when_price_crossed_sl_or_pt_for_trade = (
        new_datetime_when_price_crossed_sl_or_pt_for_trade.loc[
            new_datetime_when_price_crossed_sl_or_pt_for_trade.shift(1).lt(
                new_datetime_when_price_crossed_sl_or_pt_for_trade.index.to_series()
            )
        ]
    )

    # signals = signals.shift(-1)
    # new_datetime_when_price_crossed_sl_or_pt_for_trade = pd.Series(
    #     data=new_datetime_when_price_crossed_sl_or_pt_for_trade.values,
    #     index=signals.index.to_series().shift(1).loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index]
    # )

    new_signals = signals.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_signals = new_signals.reindex_like(signals)

    new_long_stop_loss_levels = long_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_long_stop_loss_levels = new_long_stop_loss_levels.reindex_like(signals).fillna(0)

    new_short_stop_loss_levels = short_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_short_stop_loss_levels = new_short_stop_loss_levels.reindex_like(signals).fillna(0)

    new_long_profit_target_levels = long_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_long_profit_target_levels = new_long_profit_target_levels.reindex_like(signals).fillna(0)

    new_short_profit_target_levels = short_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_short_profit_target_levels = new_short_profit_target_levels.reindex_like(signals).fillna(0)

    new_long_limit_prices = long_limit_prices.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_long_limit_prices = new_long_limit_prices.reindex_like(signals)

    new_short_limit_prices = short_limit_prices.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_short_limit_prices = new_short_limit_prices.reindex_like(signals)

    ##############

    new_signals.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    new_signals = new_signals.ffill().fillna(0)

    new_stop_loss_levels = new_long_stop_loss_levels.add(new_short_stop_loss_levels).replace(0, np.nan)
    new_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    new_stop_loss_levels = new_stop_loss_levels.ffill().replace(0, np.nan)

    new_profit_target_levels = new_long_profit_target_levels.add(new_short_profit_target_levels).replace(0, np.nan)
    new_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    new_profit_target_levels = new_profit_target_levels.ffill().replace(0, np.nan)

    new_long_limit_prices.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    new_long_limit_prices = new_long_limit_prices.ffill().replace(0, np.nan)

    new_short_limit_prices.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade] = 0
    new_short_limit_prices = new_short_limit_prices.ffill().replace(0, np.nan)

    return dict(
        forecasts=new_signals,
        stop_loss_levels_after_slpt=new_stop_loss_levels,
        profit_target_levels_after_slpt=new_profit_target_levels,
        long_limit_prices_after_slpt=new_long_limit_prices,
        short_limit_prices_after_slpt=new_short_limit_prices,
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

    price_bars = dbFuturesSimData().get_backadjusted_futures_price('CL')
    price_bars = price_bars.loc[price_bars['FINAL'] != 0.0]

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

    orion_trades = orion(price_bars)
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



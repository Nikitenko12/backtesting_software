import pandas as pd
import numpy as np

from systems.stage import SystemStage
from systems.system_cache import input, diagnostic, output


class StopLossProfitTarget(SystemStage):
    """
    This stage goes directly after the rules stage
    """
    @output
    def get_signals_after_stop_loss_and_profit_target(self):
        updated_signals_dict = apply_stop_loss_and_profit_target_to_signals(
            prices=self.prices,
            signals=self.signals,
            long_stop_loss_levels=self.long_stop_loss_levels,
            short_stop_loss_levels=self.short_stop_loss_levels,
            long_profit_target_levels=self.long_profit_target_levels,
            short_profit_target_levels=self.short_profit_target_levels,
        )

        return updated_signals_dict

    @property
    def prices(self) -> pd.DataFrame:
        return self.rules.data[1]

    @property
    def rules(self):
        return self.parent.rules

    @input
    def signals_dict(self):
        return self.rules.get_raw_forecast()

    @property
    def signals(self):
        return self.signals_dict['signals']

    @property
    def long_stop_loss_levels(self):
        return self.signals_dict['long_stop_loss_prices']

    @property
    def long_profit_target_levels(self):
        return self.signals_dict['long_profit_taker']

    @property
    def short_stop_loss_levels(self):
        return self.signals_dict['short_stop_loss_prices']

    @property
    def short_profit_target_levels(self):
        return self.signals_dict['short_profit_taker']


def apply_stop_loss_and_profit_target_to_signals(
    prices: pd.DataFrame,
    signals: pd.Series,
    long_stop_loss_levels: pd.Series,
    short_stop_loss_levels: pd.Series,
    long_profit_target_levels: pd.Series,
    short_profit_target_levels: pd.Series,
) -> dict:

    signals = signals.shift(1)

    # get index where prices crossed stop loss and profit target for each trade
    datetime_when_price_crossed_sl_or_pt_for_trade = pd.Series(None).reindex_like(signals.loc[signals != 0])
    for dt, signal in (signals.loc[signals != 0]).items():
        datetime_when_price_crossed_sl_or_pt_for_trade.loc[dt] = (
            (prices.shift(-1).loc[dt:, 'HIGH'].ge(long_profit_target_levels.loc[dt]) | prices.shift(-1).loc[dt:, 'LOW'].le(long_stop_loss_levels.loc[dt])).idxmax() if signal > 0 else (
                (prices.shift(-1).loc[dt:, 'LOW'].le(short_profit_target_levels.loc[dt]) | prices.shift(-1).loc[dt:, 'HIGH'].ge(short_stop_loss_levels.loc[dt])).idxmax()
            )
        )

    # at a certain date, get first trade (entry) where you would be in the market
    # get, for a certain signal, whether the prices hit stop loss or profit target on the previous signal's entry
    new_datetime_when_price_crossed_sl_or_pt_for_trade = datetime_when_price_crossed_sl_or_pt_for_trade.copy()
    for i in range(1, len(datetime_when_price_crossed_sl_or_pt_for_trade)):
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

    new_signals = signals.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_signals = new_signals.reindex_like(signals)

    new_long_stop_loss_levels = long_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_long_stop_loss_levels = new_long_stop_loss_levels.reindex_like(signals)

    new_short_stop_loss_levels = short_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_short_stop_loss_levels = new_short_stop_loss_levels.reindex_like(signals)

    new_long_profit_target_levels = long_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_long_profit_target_levels = new_long_profit_target_levels.reindex_like(signals)

    new_short_profit_target_levels = short_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.index].copy()
    new_short_profit_target_levels = new_short_profit_target_levels.reindex_like(signals)

    new_signals.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.values] = 0
    new_signals = new_signals.ffill()

    new_stop_loss_levels = new_long_stop_loss_levels.replace(0, np.nan).add(new_short_stop_loss_levels).replace(0, np.nan)
    new_stop_loss_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.values] = 0
    new_stop_loss_levels = new_long_stop_loss_levels.ffill().replace(0, np.nan)

    new_profit_target_levels = new_long_profit_target_levels.replace(0, np.nan).add(new_short_profit_target_levels).replace(0, np.nan)
    new_profit_target_levels.loc[new_datetime_when_price_crossed_sl_or_pt_for_trade.values] = 0
    new_profit_target_levels = new_profit_target_levels.ffill().replace(0, np.nan)

    return dict(
        forecasts=new_signals,
        stop_loss_levels_after_slpt=new_stop_loss_levels,
        profit_target_levels_after_slpt=new_profit_target_levels,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf

    from private.systems.orion.rules.orion import orion

    big_price_bars = \
    pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 30.csv', index_col=[0], parse_dates=True)[
        ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='CLOSE'))
    big_price_bars = big_price_bars.reindex(big_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))
    small_price_bars = \
    pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 5.csv', index_col=[0], parse_dates=True)[
        ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='CLOSE'))
    small_price_bars = small_price_bars.reindex(small_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))

    orion_trades = orion(big_price_bars, small_price_bars)
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
    #     small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", CLOSE="Close")),
    #     type='candle',
    #     show_nontrading=False,
    #     addplot=apds,
    # )

    new_orion_trades = apply_stop_loss_and_profit_target_to_signals(
        prices=small_price_bars,
        signals=signals,
        long_stop_loss_levels=orion_trades['long_stop_loss_prices'],
        short_stop_loss_levels=orion_trades['short_stop_loss_prices'],
        long_profit_target_levels=orion_trades['long_profit_taker'],
        short_profit_target_levels=orion_trades['short_profit_taker'],
    )
    new_signals = new_orion_trades['forecasts']

    new_apds = [
        mpf.make_addplot(small_price_bars['LOW'].where(new_signals > 0, np.nan), type='scatter', marker='^'),
        mpf.make_addplot(small_price_bars['HIGH'].where(new_signals < 0, np.nan), type='scatter', marker='v'),
        mpf.make_addplot(new_orion_trades['stop_loss_levels_after_slpt'], type='line'),
        mpf.make_addplot(new_orion_trades['profit_target_levels_after_slpt'], type='line'),
    ]
    mpf.plot(
        small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", CLOSE="Close")),
        type='candle',
        show_nontrading=False,
        addplot=new_apds,
    )



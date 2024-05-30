"""
If current 30min candle closes above the highest_of(10 previous 30min candles' high), then do the following:
    wait until current 5min candle's price is between 30min candle before the previous one's low and 30min candle before the previous one's high, then do the following:
        wait until current 5min candle's close is above previous Rachael T Fractal, then do the following:
            create buy limit order with limit price equal to the high of the lowest candle since you began looking for setup
            attach stop loss for order with stop price equal to the low of the lowest candle since you began looking for setup
            attach profit target 2:1 in relation to stop loss and limit price
"""
import pandas as pd
import numpy as np

VERY_BIG_NUMBER = 99999999999999999


def orion(big_price_bars: pd.DataFrame, small_price_bars: pd.DataFrame, setup_lookback=10, rr=2.0):

    may_we_look_for_long_setup = look_for_long_setup(big_price_bars, lookback=setup_lookback)
    may_we_look_for_short_setup = look_for_short_setup(big_price_bars, lookback=setup_lookback)

    long_zone = big_price_bars.shift(1).loc[may_we_look_for_long_setup.shift(1).fillna(False), :].reindex_like(small_price_bars, method='ffill')
    short_zone = big_price_bars.shift(1).loc[may_we_look_for_short_setup.shift(1).fillna(False), :].reindex_like(small_price_bars, method='ffill')

    small_price_bar_in_long_zone = (long_zone['LOW'] < small_price_bars['LOW']) & (small_price_bars['LOW'] < long_zone['HIGH'])
    small_price_bar_in_short_zone = (short_zone['HIGH'] > small_price_bars['HIGH']) & (small_price_bars['HIGH'] > short_zone['LOW'])

    datetime = pd.Series(list(small_price_bars.index), index=small_price_bars.index)
    eod = pd.Series(False).reindex_like(datetime).fillna(False)
    eod.iloc[:-1] = datetime.apply(lambda x: x.date()).iloc[:-1].lt(datetime.shift(-1).iloc[:-1].apply(lambda x: x.date()))     # FIXME add sessions, this is incorrect since sessions start the day before
    """
    eod.iloc[:-1] = datetime['SESSION'].iloc[:-1].lt(datetime['SESSION'].shift(-1).iloc[:-1])
    """
    stop_looking_for_long_setup = (small_price_bars['CLOSE'] < long_zone['LOW']) | eod
    stop_looking_for_short_setup = (small_price_bars['CLOSE'] > short_zone['HIGH']) | eod

    fractals = rachel_t_fractals(small_price_bars)
    long_fractals = fractals < 0
    short_fractals = fractals > 0
    long_fractal_prices = small_price_bars.loc[long_fractals, 'HIGH'].reindex_like(small_price_bars['HIGH'], method='ffill').shift(1)
    short_fractal_prices = small_price_bars.loc[short_fractals, 'LOW'].reindex_like(small_price_bars['LOW'], method='ffill').shift(1)

    long_setup = small_price_bars['CLOSE'] > long_fractal_prices
    short_setup = small_price_bars['CLOSE'] < short_fractal_prices

    long_setup_groups = small_price_bars.loc[small_price_bar_in_long_zone, 'HIGH'].reindex_like(small_price_bars['HIGH'], method='ffill')
    short_setup_groups = small_price_bars.loc[small_price_bar_in_short_zone, 'LOW'].reindex_like(small_price_bars['LOW'], method='ffill')

    lowest_low_since_began_looking_for_setup = small_price_bars.loc[
        small_price_bars['LOW'] == small_price_bars['LOW'].groupby(long_setup_groups).cummin(), 'LOW'
    ]
    highest_high_since_began_looking_for_setup = small_price_bars.loc[
        small_price_bars['HIGH'] == small_price_bars['HIGH'].groupby(short_setup_groups).cummax(), 'HIGH'
    ]

    looking_for_long_setup = may_we_look_for_long_setup.reindex_like(small_price_bars['CLOSE']).mask(
        stop_looking_for_long_setup, False
    ).ffill()
    looking_for_long_setup = looking_for_long_setup & small_price_bar_in_long_zone
    looking_for_long_setup.loc[~looking_for_long_setup] = np.nan
    looking_for_long_setup = looking_for_long_setup.mask(stop_looking_for_long_setup, False).ffill()

    looking_for_short_setup = may_we_look_for_short_setup.reindex_like(small_price_bars['CLOSE']).mask(
        stop_looking_for_short_setup, False
    ).ffill()
    looking_for_short_setup = looking_for_short_setup & small_price_bar_in_short_zone
    looking_for_short_setup.loc[~looking_for_short_setup] = np.nan
    looking_for_short_setup = looking_for_short_setup.mask(stop_looking_for_short_setup, False).ffill()

    long_signals = looking_for_long_setup & long_setup
    short_signals = looking_for_short_setup & short_setup

    signals = long_signals.astype(int) - short_signals.astype(int)

    long_limit_prices = small_price_bars.loc[lowest_low_since_began_looking_for_setup.index, 'HIGH'].reindex_like(
        small_price_bars['HIGH'], method='ffill'
    ).where(signals > 0, np.nan)
    short_limit_prices = small_price_bars.loc[highest_high_since_began_looking_for_setup.index, 'LOW'].reindex_like(
        small_price_bars['HIGH'], method='ffill'
    ).where(signals < 0, np.nan)

    long_stop_loss_prices = lowest_low_since_began_looking_for_setup.reindex_like(
        small_price_bars['LOW'], method='ffill'
    ).where(signals > 0, np.nan)
    short_stop_loss_prices = highest_high_since_began_looking_for_setup.reindex_like(
        small_price_bars['HIGH'], method='ffill'
    ).where(signals < 0, np.nan)

    long_profit_taker = (long_limit_prices + rr * (long_limit_prices - long_stop_loss_prices)).where(signals > 0, np.nan)
    short_profit_taker = (short_limit_prices - rr * (short_stop_loss_prices - short_limit_prices)).where(signals < 0, np.nan)

    return_dict = dict(
        signals=signals,
        long_limit_prices=long_limit_prices,
        short_limit_prices=short_limit_prices,
        long_stop_loss_prices=long_stop_loss_prices,
        short_stop_loss_prices=short_stop_loss_prices,
        long_profit_taker=long_profit_taker,
        short_profit_taker=short_profit_taker,
    )

    return return_dict


def rachel_t_fractals(price_bars: pd.DataFrame):
    def is_regular_fractal(bars: pd.DataFrame) -> pd.Series:
        high = bars['HIGH']
        low = bars['LOW']

        bull = pd.Series(False).reindex_like(high).fillna(False)
        bear = bull.copy()

        bull.iloc[4:] = ((
            high.shift(4).fillna(0) < high.shift(3).fillna(0)
        ) & (
            high.shift(3).fillna(0) < high.shift(2).fillna(0)
        ) & (
            high.shift(2).fillna(0) > high.shift(1).fillna(0)
        ) & (
            high.shift(1).fillna(0) > high
        )).iloc[4:]
        bear[4:] = ((
            low.shift(4).fillna(0) > low.shift(3).fillna(0)
        ) & (
            low.shift(3).fillna(0) > low.shift(2).fillna(0)
        ) & (
            low.shift(2).fillna(0) < low.shift(1).fillna(0)
        ) & (
            low.shift(1).fillna(0) < low
        )).iloc[4:]

        ret = bull.astype(int) - bear.astype(int)
        return ret

    def is_bw_fractal(bars: pd.DataFrame):
        high = bars['HIGH']
        low = bars['LOW']

        bull = pd.Series(False).reindex_like(high).fillna(False)
        bear = bull.copy()

        bull[4:] = ((
            high.shift(4).fillna(0) < high.shift(2).fillna(0)
        ) & (
            high.shift(3).fillna(0) <= high.shift(2)
        ) & (
            high.shift(2).fillna(0) >= high.shift(1).fillna(0)
        ) & (
            high.shift(2).fillna(0) > high
        )).iloc[4:]
        bear[4:] = ((
            low.shift(4).fillna(0) > low.shift(2).fillna(0)
        ) & (
            low.shift(3).fillna(0) >= low.shift(2)
        ) & (
            low.shift(2).fillna(0) <= low.shift(1).fillna(0)
        ) & (
            low.shift(2).fillna(0) < low
        )).iloc[4:]

        ret = 1 if bull else (-1 if bear else 0)
        return ret

    filteredtopf = is_regular_fractal(price_bars) > 0
    filteredbotf = is_regular_fractal(price_bars) < 0

    high2 = price_bars['HIGH'].shift(2)
    low2 = price_bars['LOW'].shift(2)

    def hhll(bars: pd.DataFrame) -> pd.DataFrame:
        valuewhen_h2 = high2.where(filteredtopf)
        valuewhen_l2 = low2.where(filteredbotf)

        hh = pd.Series(False).reindex_like(valuewhen_h2).fillna(False)
        lh = hh.copy(); hl = hh.copy(); ll = hh.copy()

        hh.iloc[2:] = ((valuewhen_h2.shift(1).fillna(0) < valuewhen_h2) & (valuewhen_h2.shift(2).fillna(0) < valuewhen_h2)).iloc[2:]
        lh.iloc[2:] = ((valuewhen_h2.shift(1).fillna(0) > valuewhen_h2) & (valuewhen_h2.shift(2).fillna(0) > valuewhen_h2)).iloc[2:]
        hl.iloc[2:] = ((valuewhen_l2.shift(1).fillna(0) < valuewhen_l2) & (valuewhen_l2.shift(2).fillna(0) < valuewhen_l2)).iloc[2:]
        ll.iloc[2:] = ((valuewhen_l2.shift(1).fillna(0) > valuewhen_l2) & (valuewhen_l2.shift(2).fillna(0) > valuewhen_l2)).iloc[2:]

        hh = hh.reindex_like(filteredtopf).fillna(False)
        lh = lh.reindex_like(filteredtopf).fillna(False)
        hl = hl.reindex_like(filteredbotf).fillna(False)
        ll = ll.reindex_like(filteredbotf).fillna(False)

        return pd.DataFrame(dict(hh=hh, lh=lh, hl=hl, ll=ll))

    def isTFFractal(mode: int, bars: pd.DataFrame, tf: int) -> pd.Series:
        hhll_ret = hhll(bars)
        hh = hhll_ret['hh']
        ll = hhll_ret['ll']

        valuewhen_hh = high2.where(hh, -VERY_BIG_NUMBER).ge(bars['HIGH'].shift(tf), 0)
        valuewhen_ll = low2.where(ll, VERY_BIG_NUMBER).le(bars['LOW'].shift(tf), 0)
        ret = valuewhen_hh.astype(int) if mode == 1 else (valuewhen_ll if mode == -1 else pd.Series(False).reindex_like(valuewhen_hh).fillna(False))
        ret.reindex_like(hhll_ret).fillna(False)

        return ret

    tf = 240
    higherhhigh = isTFFractal(1, price_bars, tf).where(hhll(price_bars)['hh']).reindex_like(price_bars['CLOSE']).fillna(False)  # Maroon above bar, offset=-2
    lowerllow = isTFFractal(-1, price_bars, tf).where(hhll(price_bars)['ll']).reindex_like(price_bars['CLOSE']).fillna(False)   # Green below bar, offset=-2

    """
    
    plotshape(ShowTimeFractals1 ? higherhhigh : na, title='Timed Top Fractals', style=shape.square, location=location.abovebar, color=maroon, text="[TH]", offset=-2)
    plotshape(ShowTimeFractals1 ? lowerllow : na, title='Timed Bottom Fractals', style=shape.square, location=location.belowbar, color=green, text="[TL]", offset=-2)
    
    """

    show_time_fractals = False
    istop = higherhhigh if show_time_fractals else filteredtopf
    isbot = lowerllow if show_time_fractals else filteredbotf
    topcount = bars_since(istop)
    botcount = bars_since(isbot)

    zigzag_h = high2.where(istop & (topcount.shift(1).fillna(0) > botcount.shift(1).fillna(0)), np.nan).ffill()
    zigzag_l = low2.where(isbot & (topcount.shift(1).fillna(0) < botcount.shift(1).fillna(0)), np.nan).ffill()
    zigzag = zigzag_h.add(zigzag_l, np.nan).ffill()

    x = zigzag.loc[(zigzag != zigzag.shift(1))].shift(4).reindex_like(zigzag).ffill()
    a = zigzag.loc[(zigzag != zigzag.shift(1))].shift(3).reindex_like(zigzag).ffill()
    b = zigzag.loc[(zigzag != zigzag.shift(1))].shift(2).reindex_like(zigzag).ffill()
    c = zigzag.loc[(zigzag != zigzag.shift(1))].shift(1).reindex_like(zigzag).ffill()
    d = zigzag.loc[(zigzag != zigzag.shift(1))].reindex_like(zigzag).ffill()

    xab = pd.Series(np.nan).reindex_like(x).fillna(np.nan)
    xad = xab.copy(); abc = xab.copy(); bcd = xab.copy()

    xab[4:] = (abs(b-a)/abs(x-a))[4:]
    xad[4:] = (abs(a-d)/abs(x-a))[4:]
    abc[4:] = (abs(b-c)/abs(a-b))[4:]
    bcd[4:] = (abs(c-d)/abs(b-c))[4:]

    xab.loc[(~np.isfinite(xab)) | (xab == 0.0)] = np.nan
    xad.loc[(~np.isfinite(xad)) | (xad == 0.0)] = np.nan
    abc.loc[(~np.isfinite(abc)) | (abc == 0.0)] = np.nan
    bcd.loc[(~np.isfinite(bcd)) | (bcd == 0.0)] = np.nan

    # xab = xab.ffill().fillna(0)
    # xad = xad.ffill().fillna(0)
    # abc = abc.ffill().fillna(0)
    # bcd = bcd.ffill().fillna(0)

    def isBat(_mode):
        _xab = (xab >= 0.382) & (xab <= 0.5)
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 1.618) & (bcd <= 2.618)
        _xad = xad <= 0.886
        return _xab & _abc & _bcd & _xad & ((d < c) if _mode == 1 else (d > c))

    def isAltBat(_mode): 
        _xab = xab <= 0.382
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 2.0) & (bcd <= 3.618)
        _xad = xad <= 1.13
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)

    def isButterfly(_mode):
        _xab = xab <= 0.786
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 1.618) & (bcd <= 2.618)
        _xad = (xad >= 1.27) & (xad <= 1.618)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isABCD(_mode):
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 1.13) & (bcd <= 2.618)
        return _abc & _bcd & (d < c if _mode == 1 else d > c)
    
    def isGartley(_mode):
        _xab = (xab >= 0.5) & (xab <= 0.618 // 0.618)
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 1.13) & (bcd <= 2.618)
        _xad = (xad >= 0.75) & (xad <= 0.875 // 0.786)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isCrab(_mode):
        _xab = (xab >= 0.75) & (xab <= 0.875 // 0.886)
        _abc = (abc >= 0.382) & (abc <= 0.886)
        _bcd = (bcd >= 2.0) & (bcd <= 3.618)
        _xad = (xad >= 1.5) & (xad <= 1.625 // 1.618)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isShark(_mode):
        _xab = (xab >= 0.5) & (xab <= 0.875 // 0.886)
        _abc = (abc >= 1.13) & (abc <= 1.618)
        _bcd = (bcd >= 1.27) & (bcd <= 2.24)
        _xad = (xad >= 0.88) & (xad <= 1.13)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def is5o(_mode):
        _xab = (xab >= 1.13) & (xab <= 1.618)
        _abc = (abc >= 1.618) & (abc <= 2.24)
        _bcd = (bcd >= 0.5) & (bcd <= 0.625)
        _xad = (xad >= 0.0) & (xad <= 0.236)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isWolf(_mode):
        _xab = (xab >= 1.27) & (xab <= 1.618)
        _abc = (abc >= 0) & (abc <= 5)
        _bcd = (bcd >= 1.27) & (bcd <= 1.618)
        _xad = (xad >= 0.0) & (xad <= 5)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isHnS(_mode):
        _xab = (xab >= 2.0) & (xab <= 10)
        _abc = (abc >= 0.90) & (abc <= 1.1)
        _bcd = (bcd >= 0.236) & (bcd <= 0.88)
        _xad = (xad >= 0.90) & (xad <= 1.1)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isConTria(_mode):
        _xab = (xab >= 0.382) & (xab <= 0.618)
        _abc = (abc >= 0.382) & (abc <= 0.618)
        _bcd = (bcd >= 0.382) & (bcd <= 0.618)
        _xad = (xad >= 0.236) & (xad <= 0.764)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)
    
    def isExpTria(_mode):
        _xab = (xab >= 1.236) & (xab <= 1.618)
        _abc = (abc >= 1.000) & (abc <= 1.618)
        _bcd = (bcd >= 1.236) & (bcd <= 2.000)
        _xad = (xad >= 2.000) & (xad <= 2.236)
        return _xab & _abc & _bcd & _xad & (d < c if _mode == 1 else d > c)

    fractals = pd.Series(0).reindex_like(price_bars['CLOSE']).fillna(0)
    fractals.loc[
        isABCD(-1) | isBat(-1) | isAltBat(-1) | isButterfly(-1) | isGartley(-1) | isCrab(-1) | isShark(-1) | is5o(-1) | isWolf(-1) | isHnS(-1) | isConTria(-1) | isExpTria(-1)
    ] = -1
    fractals.loc[
        isABCD(1) | isBat(1) | isAltBat(1) | isButterfly(1) | isGartley(1) | isCrab(1) | isShark(1) | is5o(1) | isWolf(1) | isHnS(1) | isConTria(1) | isExpTria(1)
    ] = 1

    return fractals


def bars_since(series: pd.Series):
    happened_this_bar = series.astype(int)
    happened_this_bar_cumsum = happened_this_bar.cumsum()
    barssince = happened_this_bar_cumsum.cummax().groupby(by=happened_this_bar_cumsum).cumcount()
    barssince.loc[:happened_this_bar.index[series][0]] = 0

    return barssince


def look_for_long_setup(large_price_bars: pd.DataFrame, lookback: int) -> pd.Series:
    return large_price_bars['CLOSE'].gt(large_price_bars['HIGH'].rolling(lookback).max().shift(1))


def look_for_short_setup(large_price_bars: pd.DataFrame, lookback: int) -> pd.Series:
    return large_price_bars['CLOSE'].lt(large_price_bars['LOW'].rolling(lookback).max().shift(1))


if __name__ == "__main__":
    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf

    big_price_bars = pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 30.csv', index_col=[0], parse_dates=True)[
            ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='CLOSE'))
    big_price_bars = big_price_bars.reindex(big_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))
    small_price_bars = pd.read_csv(get_resolved_pathname('data') + '/COMEX_GC1!, 5.csv', index_col=[0], parse_dates=True)[
            ['open', 'high', 'low', 'close']].rename(columns=dict(open="OPEN", high='HIGH', low='LOW', close='CLOSE'))
    small_price_bars = small_price_bars.reindex(small_price_bars.index.astype(pd.DatetimeTZDtype(tz='EST')))

    orion_trades = orion(big_price_bars, small_price_bars)
    signals = orion_trades['signals']

    apds = [
        mpf.make_addplot(small_price_bars['LOW'].where(signals > 0, np.nan), type='scatter', marker='^'),
        mpf.make_addplot(small_price_bars['HIGH'].where(signals < 0, np.nan), type='scatter', marker='v'),
        mpf.make_addplot(orion_trades['long_stop_loss_prices'], type='line'),
        mpf.make_addplot(orion_trades['long_profit_taker'], type='line'),
        mpf.make_addplot(orion_trades['short_stop_loss_prices'], type='line'),
        mpf.make_addplot(orion_trades['short_profit_taker'], type='line'),
    ]
    mpf.plot(
        small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", CLOSE="Close")),
        type='candle',
        show_nontrading=False,
        addplot=apds,
    )

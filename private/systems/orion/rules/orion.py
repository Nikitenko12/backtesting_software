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

from sysobjects.sessions import Session
from private.systems.orion.rawdata.rawdata import apply_sessions_to_aggregated_data

VERY_BIG_NUMBER = 99999999999999999


def orion(minute_bars: pd.DataFrame, sessions: Session, big_timeframe='30T', small_timeframe='5T', setup_lookback=10, rr=2.0):
    big_price_bars = minute_bars.resample(big_timeframe).agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    big_price_bars = apply_sessions_to_aggregated_data(big_price_bars, sessions)

    small_price_bars = minute_bars.resample(small_timeframe).agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    small_price_bars = apply_sessions_to_aggregated_data(small_price_bars, sessions)

    print("******* Calculating supply and demand zones *******")
    may_we_look_for_long_setup = look_for_long_setup(big_price_bars, lookback=setup_lookback)
    may_we_look_for_short_setup = look_for_short_setup(big_price_bars, lookback=setup_lookback)

    demand_zones = big_price_bars.shift(1).loc[may_we_look_for_long_setup.shift(1).fillna(False), ['HIGH', 'LOW']]
    supply_zones = big_price_bars.shift(1).loc[may_we_look_for_short_setup.shift(1).fillna(False), ['HIGH', 'LOW']]

    when_price_hit_which_demand_zone = pd.Series([[]] * len(small_price_bars.index), index=small_price_bars.index)
    when_price_hit_which_supply_zone = when_price_hit_which_demand_zone.copy()

    print("******* Checking when prices entered demand zones *******")
    for zone_dt, zone in demand_zones.iterrows():
        when_price_exits_zone_first = big_price_bars.loc[zone_dt:, 'OPEN'].gt(zone.HIGH)
        dt_when_price_exits_zone_first = pd.NaT if not when_price_exits_zone_first.any() else when_price_exits_zone_first.idxmax()

        if dt_when_price_exits_zone_first is not pd.NaT:
            dt_when_to_cancel_zone = small_price_bars.loc[dt_when_price_exits_zone_first:, 'FINAL'].lt(zone.LOW)
            dt_when_to_cancel_zone = small_price_bars.index[-1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

            did_price_hit_this_demand_zone = (
                small_price_bars.loc[dt_when_price_exits_zone_first:dt_when_to_cancel_zone, 'LOW'].gt(zone.LOW) & (
                    small_price_bars.loc[dt_when_price_exits_zone_first:dt_when_to_cancel_zone, 'LOW'].lt(zone.HIGH)
                )
            )
            did_price_hit_this_demand_zone = did_price_hit_this_demand_zone.loc[did_price_hit_this_demand_zone]
            dt_when_price_hit_this_demand_zone = did_price_hit_this_demand_zone.index

            for idx in dt_when_price_hit_this_demand_zone:
                when_price_hit_which_demand_zone.loc[idx].append((zone_dt, dt_when_price_exits_zone_first, dt_when_to_cancel_zone))

    print("******* Checking when prices entered supply zones *******")
    for zone_dt, zone in supply_zones.iterrows():
        when_price_exits_zone_first = big_price_bars.loc[zone_dt:, 'OPEN'].lt(zone.LOW)
        dt_when_price_exits_zone_first = pd.NaT if not when_price_exits_zone_first.any() else when_price_exits_zone_first.idxmax()

        if dt_when_price_exits_zone_first is not pd.NaT:
            dt_when_to_cancel_zone = small_price_bars.loc[dt_when_price_exits_zone_first:, 'FINAL'].gt(zone.HIGH)
            dt_when_to_cancel_zone = small_price_bars.index[-1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

            did_price_hit_this_supply_zone = (
                small_price_bars.loc[dt_when_price_exits_zone_first:dt_when_to_cancel_zone, 'LOW'].gt(zone.LOW) & (
                    small_price_bars.loc[dt_when_price_exits_zone_first:dt_when_to_cancel_zone, 'LOW'].lt(zone.HIGH)
                )
            )
            did_price_hit_this_supply_zone = did_price_hit_this_supply_zone.loc[did_price_hit_this_supply_zone]
            dt_when_price_hit_this_supply_zone = did_price_hit_this_supply_zone.index

            for idx in dt_when_price_hit_this_supply_zone:
                when_price_hit_which_supply_zone.loc[idx].append((zone_dt, dt_when_price_exits_zone_first, dt_when_to_cancel_zone))

    when_price_hit_which_demand_zone = when_price_hit_which_demand_zone.loc[[len(x) > 0 for x in when_price_hit_which_demand_zone]]
    when_price_hit_which_supply_zone = when_price_hit_which_supply_zone.loc[[len(x) > 0 for x in when_price_hit_which_supply_zone]]

    print("******* Calculating fractals *******")
    fractals = rachel_t_fractals(small_price_bars)
    long_fractals = fractals < 0
    short_fractals = fractals > 0
    long_fractal_prices = small_price_bars.loc[long_fractals, 'HIGH'].reindex_like(small_price_bars['HIGH'])
    short_fractal_prices = small_price_bars.loc[short_fractals, 'LOW'].reindex_like(small_price_bars['LOW'])

    long_signals = pd.Series(False).reindex_like(small_price_bars['FINAL'])
    short_signals = long_signals.copy()

    long_limit_prices = pd.Series(np.nan).reindex_like(small_price_bars['FINAL'])
    short_limit_prices = long_limit_prices.copy()

    long_stop_loss_levels = long_limit_prices.copy()
    short_stop_loss_levels = long_limit_prices.copy()

    long_profit_target_levels = long_limit_prices.copy()
    short_profit_target_levels = long_limit_prices.copy()

    print("******* Checking for long setups formed in demand zones *******")
    for dt_when_demand_zone_was_hit, zones_dts in when_price_hit_which_demand_zone.items():
        for zone_dt, dt_when_price_exits_zone_first, dt_when_to_cancel_zone in zones_dts:
            print(f'*** Demand zone formed at {zone_dt}, first exited at {dt_when_price_exits_zone_first} and canceled at {dt_when_to_cancel_zone}, was hit at ' + (
                f'{dt_when_demand_zone_was_hit}. Checking for setups ***')
            )
            when_long_setup_happened = (
                small_price_bars.loc[dt_when_demand_zone_was_hit:dt_when_to_cancel_zone, 'FINAL'].gt(
                    long_fractal_prices.shift(1).ffill()[dt_when_demand_zone_was_hit:dt_when_to_cancel_zone]
                )
            )
            dt_when_long_setup_happened = pd.NaT if not when_long_setup_happened.any() else when_long_setup_happened.idxmax()

            if dt_when_long_setup_happened is not pd.NaT:
                print(f'* Long setup happened at {dt_when_long_setup_happened}. Calculating limit, stop loss, and profit target prices *')
                long_signals[dt_when_long_setup_happened] = True

                idx_to_include_when_long_setup_happened = small_price_bars[dt_when_long_setup_happened:].index

                lowest_bar_since_demand_zone_was_hit = small_price_bars.loc[
                    dt_when_demand_zone_was_hit:idx_to_include_when_long_setup_happened[min(1, len(idx_to_include_when_long_setup_happened))], 'LOW'
                ].idxmin()

                limit_price = small_price_bars.loc[lowest_bar_since_demand_zone_was_hit, 'HIGH']
                stop_loss_level = small_price_bars.loc[lowest_bar_since_demand_zone_was_hit, 'LOW']
                profit_target_level = limit_price + rr * (limit_price - stop_loss_level)

                long_limit_prices[dt_when_long_setup_happened] = limit_price
                long_stop_loss_levels[dt_when_long_setup_happened] = stop_loss_level
                long_profit_target_levels[dt_when_long_setup_happened] = profit_target_level

    print("******* Checking for short setups formed in supply zones *******")
    for dt_when_supply_zone_was_hit, zones_dts in when_price_hit_which_supply_zone.items():
        for zone_dt, dt_when_price_exits_zone_first, dt_when_to_cancel_zone in zones_dts:
            print(f'*** Supply zone formed at {zone_dt}, first exited at {dt_when_price_exits_zone_first} and canceled at {dt_when_to_cancel_zone}, was hit at ' + (
                f'{dt_when_supply_zone_was_hit}. Checking for setups ***')
            )
            when_short_setup_happened = (
                small_price_bars.loc[dt_when_supply_zone_was_hit:dt_when_to_cancel_zone, 'FINAL'].lt(
                    short_fractal_prices.shift(1).ffill()[dt_when_supply_zone_was_hit:dt_when_to_cancel_zone]
                )
            )
            dt_when_short_setup_happened = pd.NaT if not when_short_setup_happened.any() else when_short_setup_happened.idxmax()

            if dt_when_short_setup_happened is not pd.NaT:
                print(f'* Short setup happened at {dt_when_short_setup_happened}. Calculating limit, stop loss, and profit target prices *')
                short_signals[dt_when_short_setup_happened] = True

                idx_to_include_when_short_setup_happened = small_price_bars[dt_when_short_setup_happened:].index

                highest_bar_since_supply_zone_was_hit = small_price_bars.loc[
                    dt_when_supply_zone_was_hit:idx_to_include_when_short_setup_happened[min(1, len(idx_to_include_when_short_setup_happened))], 'HIGH'
                ].idxmax()

                limit_price = small_price_bars.loc[highest_bar_since_supply_zone_was_hit, 'LOW']
                stop_loss_level = small_price_bars.loc[highest_bar_since_supply_zone_was_hit, 'HIGH']
                profit_target_level = limit_price - rr * (stop_loss_level - limit_price)

                short_limit_prices[dt_when_short_setup_happened] = limit_price
                short_stop_loss_levels[dt_when_short_setup_happened] = stop_loss_level
                short_profit_target_levels[dt_when_short_setup_happened] = profit_target_level

    print("******* Done calculating signals *******")
    signals = long_signals.astype(int) - short_signals.astype(int)

    return_dict = dict(
        signals=signals,
        long_limit_prices=long_limit_prices,
        short_limit_prices=short_limit_prices,
        long_stop_loss_prices=long_stop_loss_levels,
        short_stop_loss_prices=short_stop_loss_levels,
        long_profit_taker=long_profit_target_levels,
        short_profit_taker=short_profit_target_levels,
    )

    return return_dict


"""
def orion(minute_bars: pd.DataFrame, sessions: Session, big_timeframe='30T', small_timeframe='5T', setup_lookback=10, rr=2.0):
    big_price_bars = minute_bars.resample(big_timeframe).agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    big_price_bars = apply_sessions_to_aggregated_data(big_price_bars, sessions)

    small_price_bars = minute_bars.resample(small_timeframe).agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    small_price_bars = apply_sessions_to_aggregated_data(small_price_bars, sessions)

    may_we_look_for_long_setup = look_for_long_setup(big_price_bars, lookback=setup_lookback)
    may_we_look_for_short_setup = look_for_short_setup(big_price_bars, lookback=setup_lookback)

    long_zone = big_price_bars.shift(1).loc[may_we_look_for_long_setup.shift(1).fillna(False), :].reindex_like(small_price_bars, method='ffill')
    # long_zone.loc[~may_we_look_for_long_setup.shift(1).fillna(False)] = 0
    # long_zone = long_zone.reindex_like(small_price_bars, method='ffill').replace(0, np.nan)
    short_zone = big_price_bars.shift(1).loc[may_we_look_for_short_setup.shift(1).fillna(False), :].reindex_like(small_price_bars, method='ffill')
    # short_zone.loc[~may_we_look_for_short_setup.shift(1).fillna(False)] = 0
    # short_zone = long_zone.reindex_like(small_price_bars, method='ffill').replace(0, np.nan)

    small_price_bar_in_long_zone = (
        (long_zone['LOW'].lt(small_price_bars['LOW'])) & (small_price_bars['LOW'].lt(long_zone['HIGH']))
    )
    small_price_bar_in_short_zone = (
        (short_zone['HIGH'].gt(small_price_bars['HIGH'])) & (small_price_bars['HIGH'].gt(short_zone['LOW']))
    )

    datetime = small_price_bars.index.tz_convert(sessions.tzinfo).to_series()
    end_date_for_session = datetime.apply(
        lambda x: x.date() if x.time() < sessions.end_time or (
                sessions.end_time < sessions.start_time and x.time() < sessions.start_time) else
        x.date() + pd.Timedelta(1, 'D')
    )
    session_end_times = pd.Series(
        [pd.Timestamp(f'{x} {sessions.end_time}', tzinfo=sessions.tzinfo) for x in end_date_for_session],
        index=end_date_for_session.index
    )

    eod = datetime.asof(session_end_times.loc[datetime]).drop_duplicates()
    eod = pd.Series([x in eod.values for x in datetime.values], index=datetime.index)

    stop_looking_for_long_setup = (small_price_bars['FINAL'] < long_zone['LOW']) #| eod
    stop_looking_for_short_setup = (small_price_bars['FINAL'] > short_zone['HIGH']) #| eod

    fractals = rachel_t_fractals(small_price_bars)
    long_fractals = fractals < 0
    short_fractals = fractals > 0
    long_fractal_prices = small_price_bars.loc[long_fractals, 'HIGH'].reindex_like(small_price_bars['HIGH'], method='ffill').shift(1)
    short_fractal_prices = small_price_bars.loc[short_fractals, 'LOW'].reindex_like(small_price_bars['LOW'], method='ffill').shift(1)

    long_setup = small_price_bars['FINAL'] > long_fractal_prices
    short_setup = small_price_bars['FINAL'] < short_fractal_prices

    long_setup_groups = small_price_bars.loc[small_price_bar_in_long_zone, 'LOW'].reindex_like(small_price_bars['LOW'], method='ffill') ##FIXME
    short_setup_groups = small_price_bars.loc[small_price_bar_in_short_zone, 'HIGH'].reindex_like(small_price_bars['HIGH'], method='ffill')

    lowest_low_since_began_looking_for_setup = small_price_bars.loc[
        small_price_bars['LOW'] == small_price_bars['LOW'].groupby(long_setup_groups).cummin(), 'LOW'
    ]
    highest_high_since_began_looking_for_setup = small_price_bars.loc[
        small_price_bars['HIGH'] == small_price_bars['HIGH'].groupby(short_setup_groups).cummax(), 'HIGH'
    ]

    looking_for_long_setup = may_we_look_for_long_setup.reindex_like(small_price_bars['FINAL']).mask(
        stop_looking_for_long_setup, False
    ).ffill()
    looking_for_long_setup.loc[~stop_looking_for_long_setup & ~small_price_bar_in_long_zone] = pd.NA
    looking_for_long_setup.ffill(inplace=True)

    looking_for_short_setup = may_we_look_for_short_setup.reindex_like(small_price_bars['FINAL']).mask(
        stop_looking_for_short_setup, False
    ).ffill()
    looking_for_short_setup.loc[~stop_looking_for_short_setup & ~small_price_bar_in_short_zone] = pd.NA
    looking_for_short_setup.ffill(inplace=True)

    long_signals = looking_for_long_setup & long_setup
    short_signals = looking_for_short_setup & short_setup

    signals = long_signals.astype(int) - short_signals.astype(int)

    long_limit_prices = small_price_bars.loc[lowest_low_since_began_looking_for_setup.index, 'HIGH'].reindex_like(
        small_price_bars['HIGH'], method='ffill'
    ).where(signals > 0, np.nan)
    short_limit_prices = small_price_bars.loc[highest_high_since_began_looking_for_setup.index, 'LOW'].reindex_like(
        small_price_bars['LOW'], method='ffill'
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
        long_zones=long_zone,
        short_zones=short_zone,
    )

    return return_dict
"""

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
    higherhhigh = isTFFractal(1, price_bars, tf).where(hhll(price_bars)['hh']).reindex_like(price_bars['FINAL']).fillna(False)  # Maroon above bar, offset=-2
    lowerllow = isTFFractal(-1, price_bars, tf).where(hhll(price_bars)['ll']).reindex_like(price_bars['FINAL']).fillna(False)   # Green below bar, offset=-2

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

    fractals = pd.Series(0).reindex_like(price_bars['FINAL']).fillna(0)
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
    if any(happened_this_bar):
        barssince.loc[:happened_this_bar.index[series][0]] = 0

    return barssince


def look_for_long_setup(large_price_bars: pd.DataFrame, lookback: int) -> pd.Series:
    return large_price_bars['FINAL'].lt(large_price_bars['LOW'].rolling(lookback).min().shift(1))


def look_for_short_setup(large_price_bars: pd.DataFrame, lookback: int) -> pd.Series:
    return large_price_bars['FINAL'].gt(large_price_bars['HIGH'].rolling(lookback).max().shift(1))


if __name__ == "__main__":
    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf
    from sysdata.sim.db_futures_sim_data import dbFuturesSimData

    data = dbFuturesSimData()
    minute_bars = data.get_backadjusted_futures_price('CL')
    sessions = data.get_sessions_for_instrument('CL')

    orion_trades = orion(minute_bars, sessions=sessions, small_timeframe='5T', big_timeframe='30T', rr=2.5)

    signals = orion_trades['signals']

    big_price_bars = minute_bars.resample('30T').agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    big_price_bars = apply_sessions_to_aggregated_data(big_price_bars, sessions)

    small_price_bars = minute_bars.resample('5T').agg(
        {
            'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'FINAL': 'last',
            'VOLUME': 'sum',
        }
    )
    small_price_bars = apply_sessions_to_aggregated_data(small_price_bars, sessions)

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

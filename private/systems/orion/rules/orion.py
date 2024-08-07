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


def orion(minute_bars: pd.DataFrame, sessions: Session, big_timeframe='30T', small_timeframe='5T', setup_lookback=50, rr=2.0):
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

    # swings = swing_highs_lows(big_price_bars, swing_length=5)
    # order_blocks = ob(big_price_bars, swing_highs_lows=swings, close_mitigation=False).shift(1).dropna() # shift(1) because bar dt is bar open dt
    #
    # demand_zones_idx = order_blocks.loc[order_blocks.OB > 0].index
    # supply_zones_idx = order_blocks.loc[order_blocks.OB < 0].index
    #
    # demand_zones = pd.DataFrame(
    #     dict(HIGH=order_blocks.Top[demand_zones_idx].values, LOW=order_blocks.Bottom[demand_zones_idx].values),
    #     index=big_price_bars.index[order_blocks.loc[demand_zones_idx, 'CloseIndex'].astype(int)]
    # )
    # supply_zones = pd.DataFrame(
    #     dict(HIGH=order_blocks.Top[supply_zones_idx].values, LOW=order_blocks.Bottom[supply_zones_idx].values),
    #     index=big_price_bars.index[order_blocks.loc[supply_zones_idx, 'CloseIndex'].astype(int)]
    # )

    demand_zones, supply_zones, swing_highs, swing_lows = get_zones(big_price_bars)

    when_price_hit_which_demand_zone = pd.Series([list() for _ in small_price_bars.index], index=small_price_bars.index)
    when_price_hit_which_supply_zone = pd.Series([list() for _ in small_price_bars.index], index=small_price_bars.index)

    print("******* Checking when prices entered demand zones *******")
    for zone_dt, zone in demand_zones.iterrows():
        print(f'Zone from {zone_dt}')
        # input("Continue? ")
        dt_when_to_cancel_zone = small_price_bars.loc[zone_dt:, 'FINAL'].lt(zone.LOW)
        dt_when_to_cancel_zone = small_price_bars.index[-1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        did_price_hit_this_demand_zone = (
            small_price_bars.loc[zone_dt:dt_when_to_cancel_zone, 'LOW'].gt(zone.LOW) & (
                small_price_bars.loc[zone_dt:dt_when_to_cancel_zone, 'LOW'].lt(zone.HIGH)
            )
        )
        did_price_hit_this_demand_zone = did_price_hit_this_demand_zone.loc[did_price_hit_this_demand_zone]
        dt_when_price_hit_this_demand_zone = did_price_hit_this_demand_zone.index

        for idx in dt_when_price_hit_this_demand_zone:
            when_price_hit_which_demand_zone.loc[idx].append((zone_dt, dt_when_to_cancel_zone))

    print("******* Checking when prices entered supply zones *******")
    for zone_dt, zone in supply_zones.iterrows():
        dt_when_to_cancel_zone = small_price_bars.loc[zone_dt:, 'FINAL'].gt(zone.HIGH)
        dt_when_to_cancel_zone = small_price_bars.index[-1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        did_price_hit_this_supply_zone = (
            small_price_bars.loc[zone_dt:dt_when_to_cancel_zone, 'LOW'].gt(zone.LOW) & (
                small_price_bars.loc[zone_dt:dt_when_to_cancel_zone, 'LOW'].lt(zone.HIGH)
            )
        )
        did_price_hit_this_supply_zone = did_price_hit_this_supply_zone.loc[did_price_hit_this_supply_zone]
        dt_when_price_hit_this_supply_zone = did_price_hit_this_supply_zone.index

        for idx in dt_when_price_hit_this_supply_zone:
            when_price_hit_which_supply_zone.loc[idx].append((zone_dt, dt_when_to_cancel_zone))

    when_price_hit_which_demand_zone = when_price_hit_which_demand_zone.loc[[len(x) > 0 for x in when_price_hit_which_demand_zone]]
    when_price_hit_which_supply_zone = when_price_hit_which_supply_zone.loc[[len(x) > 0 for x in when_price_hit_which_supply_zone]]

    print("******* Calculating fractals *******")
    red_fractals, red_fractals_prices, green_fractals, green_fractals_prices = rachel_t_fractals(small_price_bars)

    long_signals = pd.Series(False, index=small_price_bars.index)
    short_signals = pd.Series(False, index=small_price_bars.index)

    long_limit_prices = pd.Series(np.nan, index=small_price_bars.index)
    short_limit_prices = pd.Series(np.nan, index=small_price_bars.index)

    long_stop_loss_levels = pd.Series(np.nan, index=small_price_bars.index)
    short_stop_loss_levels = pd.Series(np.nan, index=small_price_bars.index)

    long_profit_target_levels = pd.Series(np.nan, index=small_price_bars.index)
    short_profit_target_levels = pd.Series(np.nan, index=small_price_bars.index)
    
    which_demand_zone_generated_each_long_signal = pd.DataFrame(dict(zone_dt=pd.NaT, HIGH=np.nan, LOW=np.nan), index=small_price_bars.index)
    which_supply_zone_generated_each_short_signal = pd.DataFrame(dict(zone_dt=pd.NaT, HIGH=np.nan, LOW=np.nan), index=small_price_bars.index)

    print("******* Checking for long setups formed in demand zones *******")
    for dt_when_demand_zone_was_hit, zones_dts in when_price_hit_which_demand_zone.items():
        for zone_dt, dt_when_to_cancel_zone in zones_dts:
            print(f'*** Demand zone formed at {zone_dt} and canceled at {dt_when_to_cancel_zone}, was hit at ' + (
                f'{dt_when_demand_zone_was_hit}. Checking for setups ***')
            )
            # input('Continue? ')
            if dt_when_demand_zone_was_hit < small_price_bars.index[-1]:
                dt_when_to_start_to_look_for_setups = small_price_bars.loc[dt_when_demand_zone_was_hit:].index[1]
                if dt_when_to_start_to_look_for_setups < dt_when_to_cancel_zone:
                    when_long_setup_happened = (
                        small_price_bars.loc[dt_when_to_start_to_look_for_setups:dt_when_to_cancel_zone, 'FINAL'].gt(
                            red_fractals_prices.ffill()[dt_when_to_start_to_look_for_setups:dt_when_to_cancel_zone]
                        )
                    )
                    dt_when_long_setup_happened = pd.NaT if not when_long_setup_happened.any() else when_long_setup_happened.idxmax()

                    if dt_when_long_setup_happened is not pd.NaT:
                        last_valid_zone = which_demand_zone_generated_each_long_signal.zone_dt.last_valid_index()
                        if last_valid_zone is not None:
                            zone_which_generated_last_long_signal = which_demand_zone_generated_each_long_signal.loc[
                                last_valid_zone
                            ]
                        else:
                            zone_which_generated_last_long_signal = None

                        if zone_which_generated_last_long_signal is None or zone_which_generated_last_long_signal.zone_dt != zone_dt:
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

                            which_demand_zone_generated_each_long_signal.loc[dt_when_long_setup_happened, 'zone_dt'] = zone_dt
                            which_demand_zone_generated_each_long_signal.loc[dt_when_long_setup_happened, 'HIGH'] = demand_zones.loc[zone_dt, 'HIGH']
                            which_demand_zone_generated_each_long_signal.loc[dt_when_long_setup_happened, 'LOW'] = demand_zones.loc[zone_dt, 'LOW']

    print("******* Checking for short setups formed in supply zones *******")
    for dt_when_supply_zone_was_hit, zones_dts in when_price_hit_which_supply_zone.items():
        for zone_dt, dt_when_to_cancel_zone in zones_dts:
            print(f'*** Supply zone formed at {zone_dt} and canceled at {dt_when_to_cancel_zone}, was hit at ' + (
                f'{dt_when_supply_zone_was_hit}. Checking for setups ***')
            )
            if dt_when_supply_zone_was_hit < small_price_bars.index[-1]:
                dt_when_to_start_to_look_for_setups = small_price_bars.loc[dt_when_supply_zone_was_hit:].index[1]
                if dt_when_to_start_to_look_for_setups < dt_when_to_cancel_zone:
                    when_short_setup_happened = (
                        small_price_bars.loc[dt_when_to_start_to_look_for_setups:dt_when_to_cancel_zone, 'FINAL'].lt(
                            green_fractals_prices.ffill()[dt_when_to_start_to_look_for_setups:dt_when_to_cancel_zone]
                        )
                    )
                    dt_when_short_setup_happened = pd.NaT if not when_short_setup_happened.any() else when_short_setup_happened.idxmax()

                    if dt_when_short_setup_happened is not pd.NaT:
                        last_valid_zone = which_supply_zone_generated_each_short_signal.zone_dt.last_valid_index()
                        if last_valid_zone is not None:
                            zone_which_generated_last_short_signal = which_supply_zone_generated_each_short_signal.loc[
                                last_valid_zone
                            ]
                        else:
                            zone_which_generated_last_short_signal = None

                        if zone_which_generated_last_short_signal is None or zone_which_generated_last_short_signal.zone_dt != zone_dt:
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

                            which_supply_zone_generated_each_short_signal.loc[dt_when_short_setup_happened, 'zone_dt'] = zone_dt
                            which_supply_zone_generated_each_short_signal.loc[dt_when_short_setup_happened, 'HIGH'] = supply_zones.loc[zone_dt, 'HIGH']
                            which_supply_zone_generated_each_short_signal.loc[dt_when_short_setup_happened, 'LOW'] = supply_zones.loc[zone_dt, 'LOW']

    print("******* Done calculating signals *******")
    # signals = long_signals.astype(int) - short_signals.astype(int)

    return_dict = dict(
        long_signals=long_signals,
        short_signals=short_signals,
        long_limit_prices=long_limit_prices,
        short_limit_prices=short_limit_prices,
        long_stop_loss_prices=long_stop_loss_levels,
        short_stop_loss_prices=short_stop_loss_levels,
        long_profit_taker=long_profit_target_levels,
        short_profit_taker=short_profit_target_levels,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        demand_zones=demand_zones,
        supply_zones=supply_zones,
        when_price_hit_which_demand_zone=when_price_hit_which_demand_zone,
        when_price_hit_which_supply_zone=when_price_hit_which_supply_zone,
        which_demand_zone_generated_each_long_signal=which_demand_zone_generated_each_long_signal,
        which_supply_zone_generated_each_short_signal=which_supply_zone_generated_each_short_signal,
        red_fractals=red_fractals,
        green_fractals=green_fractals,
        red_fractals_prices=red_fractals_prices,
        green_fractals_prices=green_fractals_prices,
    )

    return return_dict


def rachel_t_fractals(
    price_bars: pd.DataFrame,
    showPatterns: bool = False,
    filterBW: bool = False,
    ShowHHLL: bool = False,
    ShowTimeFractals1: bool = False,
    timeframe1: str = '240 min',
    showZigZag: bool = True,
):
    def is_regular_fractal(bars: pd.DataFrame) -> pd.Series:
        high = bars['HIGH']
        low = bars['LOW']

        bull = pd.Series(False).reindex_like(high).fillna(False)
        bear = pd.Series(False).reindex_like(high).fillna(False)

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
        bear = pd.Series(False).reindex_like(high).fillna(False)

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

        ret = pd.Series(1, index=bars.index).where(bull, other=pd.Series(-1, index=bars.index).where(bear, other=0))
        return ret

    filteredtopf = is_regular_fractal(price_bars) > 0 if filterBW else is_bw_fractal(price_bars) > 0
    filteredbotf = is_regular_fractal(price_bars) < 0 if filterBW else is_bw_fractal(price_bars) < 0

    red_fractals = filteredtopf
    red_fractal_prices = price_bars.loc[red_fractals.shift(-2).fillna(False), 'HIGH'].reindex_like(red_fractals).shift(2)
    green_fractals = filteredbotf
    green_fractals_prices = price_bars.loc[green_fractals.shift(-2).fillna(False), 'LOW'].reindex_like(green_fractals).shift(2)

    high2 = price_bars['HIGH'].shift(2)
    low2 = price_bars['LOW'].shift(2)

    def hhll(bars: pd.DataFrame) -> pd.DataFrame:
        valuewhen_h2 = high2.where(filteredtopf)
        valuewhen_l2 = low2.where(filteredbotf)

        hh = pd.Series(False).reindex_like(valuewhen_h2).fillna(False)
        lh = pd.Series(False).reindex_like(valuewhen_h2).fillna(False)
        hl = pd.Series(False).reindex_like(valuewhen_h2).fillna(False)
        ll = pd.Series(False).reindex_like(valuewhen_h2).fillna(False)

        hh.iloc[2:] = ((valuewhen_h2.shift(1).fillna(0) < valuewhen_h2) & (valuewhen_h2.shift(2).fillna(0) < valuewhen_h2)).iloc[2:]
        lh.iloc[2:] = ((valuewhen_h2.shift(1).fillna(0) > valuewhen_h2) & (valuewhen_h2.shift(2).fillna(0) > valuewhen_h2)).iloc[2:]
        hl.iloc[2:] = ((valuewhen_l2.shift(1).fillna(0) < valuewhen_l2) & (valuewhen_l2.shift(2).fillna(0) < valuewhen_l2)).iloc[2:]
        ll.iloc[2:] = ((valuewhen_l2.shift(1).fillna(0) > valuewhen_l2) & (valuewhen_l2.shift(2).fillna(0) > valuewhen_l2)).iloc[2:]

        hh = hh.reindex_like(filteredtopf).fillna(False)
        lh = lh.reindex_like(filteredtopf).fillna(False)
        hl = hl.reindex_like(filteredbotf).fillna(False)
        ll = ll.reindex_like(filteredbotf).fillna(False)

        return pd.DataFrame(dict(hh=hh, lh=lh, hl=hl, ll=ll))

    def isTFFractal(mode: int, bars: pd.DataFrame, tf: str) -> pd.Series:
        hhll_ret = hhll(bars)
        hh = hhll_ret['hh']
        ll = hhll_ret['ll']

        higher_timeframe_bars = bars.resample(tf).agg(
            {
                'OPEN': 'first',
                'HIGH': 'max',
                'LOW': 'min',
                'FINAL': 'last',
                'VOLUME': 'sum',
            }
        ).iloc[-1]

        valuewhen_hh = high2.where(hh, -VERY_BIG_NUMBER).ge(higher_timeframe_bars['HIGH'], 0)
        valuewhen_ll = low2.where(ll, VERY_BIG_NUMBER).le(higher_timeframe_bars['LOW'], 0)
        ret = valuewhen_hh.astype(int) if mode == 1 else (valuewhen_ll if mode == -1 else pd.Series(False).reindex_like(valuewhen_hh).fillna(False))
        ret.reindex_like(hhll_ret).fillna(False)

        return ret

    higherhhigh = isTFFractal(1, price_bars, timeframe1).where(hhll(price_bars)['hh']).reindex_like(price_bars['FINAL']).fillna(False)  # Maroon above bar, offset=-2
    lowerllow = isTFFractal(-1, price_bars, timeframe1).where(hhll(price_bars)['ll']).reindex_like(price_bars['FINAL']).fillna(False)   # Green below bar, offset=-2

    istop = higherhhigh if ShowTimeFractals1 else filteredtopf
    isbot = lowerllow if ShowTimeFractals1 else filteredbotf
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
    xad = pd.Series(np.nan).reindex_like(x).fillna(np.nan)
    abc = pd.Series(np.nan).reindex_like(x).fillna(np.nan)
    bcd = pd.Series(np.nan).reindex_like(x).fillna(np.nan)

    xab[4:] = (abs(b-a)/abs(x-a))[4:]
    xad[4:] = (abs(a-d)/abs(x-a))[4:]
    abc[4:] = (abs(b-c)/abs(a-b))[4:]
    bcd[4:] = (abs(c-d)/abs(b-c))[4:]

    xab.loc[(~np.isfinite(xab)) | (xab == 0.0)] = np.nan
    xad.loc[(~np.isfinite(xad)) | (xad == 0.0)] = np.nan
    abc.loc[(~np.isfinite(abc)) | (abc == 0.0)] = np.nan
    bcd.loc[(~np.isfinite(bcd)) | (bcd == 0.0)] = np.nan

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

    if showPatterns:
        fractals = pd.Series(0, index=price_bars.index)
        fractals.loc[
            isABCD(-1) | isBat(-1) | isAltBat(-1) | isButterfly(-1) | isGartley(-1) | isCrab(-1) | isShark(-1) | is5o(-1) | isWolf(-1) | isHnS(-1) | isConTria(-1) | isExpTria(-1)
        ] = -1
        fractals.loc[
            isABCD(1) | isBat(1) | isAltBat(1) | isButterfly(1) | isGartley(1) | isCrab(1) | isShark(1) | is5o(1) | isWolf(1) | isHnS(1) | isConTria(1) | isExpTria(1)
        ] = 1
        fractals.loc[fractals.eq(fractals.shift(1))] = 0

    # return_df = pd.concat(
    #     [
    #         red_fractals,
    #         red_fractals_price_idxs,
    #         green_fractals,
    #         green_fractals_price_idxs,
    #     ],
    #     axis=1,
    # )

    return red_fractals, red_fractal_prices, green_fractals, green_fractals_prices


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


def swing_highs_lows(ohlc: pd.DataFrame, swing_length: int = 50) -> pd.DataFrame:
    """
    Swing Highs and Lows
    A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
    A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

    parameters:
    swing_length: int - the amount of candles to look back and forward to determine the swing high or low

    returns:
    HighLow = 1 if swing high, -1 if swing low
    Level = the level of the swing high or low
    """

    swing_length *= 2
    # set the highs to 1 if the current high is the highest high in the last 5 candles and next 5 candles
    swing_highs_lows = np.where(
        ohlc["HIGH"]
        == ohlc["HIGH"].shift(-(swing_length // 2)).rolling(swing_length).max(),
        1,
        np.where(
            ohlc["LOW"]
            == ohlc["LOW"].shift(-(swing_length // 2)).rolling(swing_length).min(),
            -1,
            np.nan,
        ),
    )

    # while True:
    #     positions = np.where(~np.isnan(swing_highs_lows))[0]
    #
    #     if len(positions) < 2:
    #         break
    #
    #     current = swing_highs_lows[positions[:-1]]
    #     next = swing_highs_lows[positions[1:]]
    #
    #     highs = ohlc["HIGH"].iloc[positions[:-1]].values
    #     lows = ohlc["LOW"].iloc[positions[:-1]].values
    #
    #     next_highs = ohlc["HIGH"].iloc[positions[1:]].values
    #     next_lows = ohlc["LOW"].iloc[positions[1:]].values
    #
    #     index_to_remove = np.zeros(len(positions), dtype=bool)
    #
    #     consecutive_highs = (current == 1) & (next == 1)
    #     index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
    #     index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)
    #
    #     consecutive_lows = (current == -1) & (next == -1)
    #     index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
    #     index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)
    #
    #     if not index_to_remove.any():
    #         break
    #
    #     swing_highs_lows[positions[index_to_remove]] = np.nan
    #
    # positions = np.where(~np.isnan(swing_highs_lows))[0]
    #
    # if len(positions) > 0:
    #     if swing_highs_lows[positions[0]] == 1:
    #         swing_highs_lows[0] = -1
    #     if swing_highs_lows[positions[0]] == -1:
    #         swing_highs_lows[0] = 1
    #     if swing_highs_lows[positions[-1]] == -1:
    #         swing_highs_lows[-1] = 1
    #     if swing_highs_lows[positions[-1]] == 1:
    #         swing_highs_lows[-1] = -1

    level = np.where(
        ~np.isnan(swing_highs_lows),
        np.where(swing_highs_lows == 1, ohlc["HIGH"], ohlc["LOW"]),
        np.nan,
    )

    return pd.concat(
        [
            pd.Series(swing_highs_lows, name="HighLow"),
            pd.Series(level, name="Level"),
        ],
        axis=1,
    )


def ob(ohlc: pd.DataFrame, swing_highs_lows: pd.DataFrame, close_mitigation: bool = False) -> pd.DataFrame:
    """
    OB - Order Blocks
    This method detects order blocks when there is a high amount of market orders exist on a price range.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

    returns:
    OB = 1 if bullish order block, -1 if bearish order block
    Top = top of the order block
    Bottom = bottom of the order block
    OBVolume = volume + 2 last volumes amounts
    Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))
    """
    swing_highs_lows = swing_highs_lows.copy()
    ohlc_len = len(ohlc)

    _open = ohlc["OPEN"].values
    _high = ohlc["HIGH"].values
    _low = ohlc["LOW"].values
    _close = ohlc["FINAL"].values
    _volume = ohlc["VOLUME"].values
    _swing_high_low = swing_highs_lows["HighLow"].values

    crossed = np.full(len(ohlc), False, dtype=bool)
    ob = np.zeros(len(ohlc), dtype=np.int32)
    top = np.zeros(len(ohlc), dtype=np.float32)
    bottom = np.zeros(len(ohlc), dtype=np.float32)
    obVolume = np.zeros(len(ohlc), dtype=np.float32)
    lowVolume = np.zeros(len(ohlc), dtype=np.float32)
    highVolume = np.zeros(len(ohlc), dtype=np.float32)
    percentage = np.zeros(len(ohlc), dtype=np.int32)
    mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
    breaker = np.full(len(ohlc), False, dtype=bool)
    closeIndexes = np.zeros(len(ohlc), dtype=np.int32)

    for i in range(ohlc_len):
        close_index = i

        # Bullish Order Block
        last_top_indices = np.where(
            (_swing_high_low == 1)
            & (np.arange(len(swing_highs_lows["HighLow"])) < close_index)
        )[0]

        if last_top_indices.size > 0:
            last_top_index = np.max(last_top_indices)
        else:
            last_top_index = None

        if last_top_index is not None:

            swing_top_price = _high[last_top_index]
            if _close[close_index] > swing_top_price and not crossed[last_top_index]:
                crossed[last_top_index] = True
                obBtm = _high[close_index - 1]
                obTop = _low[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_top_index):
                    obBtm = min(
                        _low[last_top_index + j],
                        obBtm,
                    )
                    if obBtm == _low[last_top_index + j]:
                        obTop = _high[last_top_index + j]
                    obIndex = (
                        last_top_index + j
                        if obBtm == _low[last_top_index + j]
                        else obIndex
                    )

                ob[obIndex] = 1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                        _volume[close_index]
                        + _volume[close_index - 1]
                        + _volume[close_index - 2]
                )
                lowVolume[obIndex] = _volume[close_index - 2]
                highVolume[obIndex] = _volume[close_index] + _volume[close_index - 1]
                percentage[obIndex] = (
                                          np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                                          / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                                          if np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0) != 0
                                          else 1
                                      ) * 100.0
                closeIndexes[obIndex] = close_index

    for i in range(ohlc_len):
        close_index = i
        close_price = _close[close_index]

        last_btm_indices = np.where(
            (swing_highs_lows["HighLow"] == -1)
            & (np.arange(len(swing_highs_lows["HighLow"])) < close_index)
        )[0]
        if last_btm_indices.size > 0:
            last_btm_index = np.max(last_btm_indices)
        else:
            last_btm_index = None

        if last_btm_index is not None:
            swing_btm_price = _low[last_btm_index]
            if close_price < swing_btm_price and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                obBtm = _low[close_index - 1]
                obTop = _high[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_btm_index):
                    obTop = max(_high[last_btm_index + j], obTop)
                    obBtm = (
                        _low[last_btm_index + j]
                        if obTop == _high[last_btm_index + j]
                        else obBtm
                    )
                    obIndex = (
                        last_btm_index + j
                        if obTop == _high[last_btm_index + j]
                        else obIndex
                    )

                ob[obIndex] = -1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                        _volume[close_index]
                        + _volume[close_index - 1]
                        + _volume[close_index - 2]
                )
                lowVolume[obIndex] = _volume[close_index] + _volume[close_index - 1]
                highVolume[obIndex] = _volume[close_index - 2]
                percentage[obIndex] = (
                                          np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                                          / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                                          if np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0) != 0
                                          else 1
                                      ) * 100.0
                closeIndexes[obIndex] = close_index

    ob = np.where(ob != 0, ob, np.nan)
    top = np.where(~np.isnan(ob), top, np.nan)
    bottom = np.where(~np.isnan(ob), bottom, np.nan)
    obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
    mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
    percentage = np.where(~np.isnan(ob), percentage, np.nan)
    closeIndexes = np.where(~np.isnan(ob), closeIndexes, np.nan)

    ob_series = pd.Series(ob, name="OB")
    top_series = pd.Series(top, name="Top")
    bottom_series = pd.Series(bottom, name="Bottom")
    obVolume_series = pd.Series(obVolume, name="OBVolume")
    mitigated_index_series = pd.Series(mitigated_index, name="MitigatedIndex")
    percentage_series = pd.Series(percentage, name="Percentage")
    close_index_series = pd.Series(closeIndexes, name="CloseIndex")
    
    return pd.concat(
        [
            ob_series,
            top_series,
            bottom_series,
            obVolume_series,
            mitigated_index_series,
            percentage_series,
            close_index_series,
        ],
        axis=1,
    )


def get_zones(bars: pd.DataFrame, length: int = 50):
    price_index_series = bars.index.to_series()
    supply_zones = pd.DataFrame(dict(HIGH=np.nan, LOW=np.nan, idx=pd.NaT), index=bars.index)
    demand_zones = pd.DataFrame(dict(HIGH=np.nan, LOW=np.nan, idx=pd.NaT), index=bars.index)

    open = bars.OPEN
    high = bars.HIGH
    low = bars.LOW
    close = bars.FINAL

    def atr(length: int = 14):
        true_range = pd.concat([(high - low), (high - close.shift(1)).abs()], axis=1).max(axis=1)
        true_range = pd.concat([true_range, (low - close.shift(1)).abs()], axis=1).max(axis=1)
        true_range.iloc[0] = high.iloc[0] - low.iloc[0]

        return true_range.rolling(length).mean()

    ATR = atr()

    def swings(len: int):
        upper = high.rolling(len).max()
        lower = low.rolling(len).min()

        os = pd.Series(0, index=bars.index).where(
            high.shift(len).gt(upper), other=pd.Series(1, index=bars.index).where(
                low.shift(len).lt(lower), np.nan
            )
        ).ffill().fillna(0)

        top = high.shift(len).where(os.eq(0) & os.shift(1).fillna(0).ne(0), other=0)
        btm = low.shift(len).where(os.eq(1) & os.shift(1).fillna(0).ne(1), other=0)

        return top, btm

    def ob_coord(use_max, len, n):
        min = np.inf
        max = -np.inf
        idx = 1

        ob_threshold = ATR

        # Search for highest/lowest high within the structure interval and get range
        if use_max:
            for i in range(1, len):
                if (high.iloc[n-i] - low.iloc[n-i]) < ob_threshold[n-i] * 2:
                    max = np.max([high.iloc[n-i], max])
                    min = low.iloc[n-i] if max == high.iloc[n-i] else min
                    idx = i if max == high.iloc[n-i] else idx
        else:
            for i in range(1, len):
                if (high.iloc[n-i] - low.iloc[n-i]) < ob_threshold[n-i] * 2:
                    min = np.min([low.iloc[n-i], min])
                    max = high.iloc[n-i] if min == low.iloc[n-i] else max
                    idx = i if min == low.iloc[n-i] else idx

        return dict(HIGH=max, LOW=min, idx=price_index_series.iloc[idx])

    def crossover(series1, series2):
        return series1.gt(series2) & series1.shift(1).le(series2.shift(1))

    def crossunder(series1, series2):
        return series1.lt(series2) & series1.shift(1).ge(series2.shift(1))

    top, btm = swings(length)
    itop, ibtm = swings(5)

    top_y = top
    top_x = price_index_series.shift(length)

    itop_y = itop
    itop_x = price_index_series.shift(5)
    itop_cross = True

    btm_y = btm
    btm_x = price_index_series.shift(length)

    ibtm_y = ibtm
    ibtm_x = price_index_series.shift(5)
    ibtm_cross = True

    when_to_show_bullish_obs = crossover(close, itop_y) # & itop_cross & top_y.ne(itop_y)
    dts_when_to_show_bullish_obs = when_to_show_bullish_obs.loc[when_to_show_bullish_obs].index
    for dt in dts_when_to_show_bullish_obs:
        demand_zones.loc[dt] = ob_coord(False, 5, bars.index.get_indexer([dt])[0])

    when_to_show_bearish_obs = crossunder(close, ibtm_y) # & ibtm_cross & btm_y.ne(ibtm_y)
    dts_when_to_show_bearish_obs = when_to_show_bearish_obs.loc[when_to_show_bearish_obs].index
    for dt in dts_when_to_show_bearish_obs:
        supply_zones.loc[dt] = ob_coord(True, 5, bars.index.get_indexer([dt])[0])

    return demand_zones.dropna(), supply_zones.dropna(), itop_y, ibtm_y


if __name__ == "__main__":
    from syscore.fileutils import get_resolved_pathname
    import mplfinance as mpf
    from sysdata.sim.db_futures_sim_data import dbFuturesSimData

    data = dbFuturesSimData()
    minute_bars = data.get_backadjusted_futures_price('CL')
    minute_bars = minute_bars.loc[minute_bars['FINAL'] != 0.0]
    sessions = data.get_sessions_for_instrument('CL')

    orion_trades = orion(minute_bars, sessions=sessions, small_timeframe='5T', big_timeframe='30T', rr=2.5)

    long_signals = orion_trades['long_signals']
    short_signals = orion_trades['short_signals']

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

    when_price_hit_which_demand_zone = orion_trades['when_price_hit_which_demand_zone'].reset_index()[0]
    when_price_hit_which_supply_zone = orion_trades['when_price_hit_which_supply_zone'].reset_index()[0]

    swing_highs = orion_trades['swing_highs']
    swing_highs.loc[swing_highs.eq(0)] = np.nan
    swing_highs = swing_highs.reindex_like(small_price_bars['FINAL'])

    swing_lows = orion_trades['swing_lows']
    swing_lows.loc[swing_lows.eq(0)] = np.nan
    swing_lows = swing_lows.reindex_like(small_price_bars['FINAL'])

    apds = [
        # mpf.make_addplot(small_price_bars['LOW'].where(long_signals, np.nan), type='scatter', marker='^'),
        # mpf.make_addplot(small_price_bars['HIGH'].where(short_signals, np.nan), type='scatter', marker='v'),
        mpf.make_addplot(swing_highs, type='scatter', color='purple', marker='x'),
        mpf.make_addplot(swing_lows, type='scatter', color='yellow', marker='x'),
        # mpf.make_addplot(orion_trades['red_fractals_prices'], type='scatter', color='red', marker='v'),
        # mpf.make_addplot(orion_trades['green_fractals_prices'], type='scatter', color='green', marker='^'),
        # mpf.make_addplot(orion_trades['long_stop_loss_prices'], type='line'),
        # mpf.make_addplot(orion_trades['long_profit_taker'], type='line'),
        # mpf.make_addplot(orion_trades['short_stop_loss_prices'], type='line'),
        # mpf.make_addplot(orion_trades['short_profit_taker'], type='line'),
    ]

    demand_zones_dts = pd.Series([x[i] for x in when_price_hit_which_demand_zone.values for i in range(len(x))]).drop_duplicates()
    for dt, demand_zone in orion_trades['demand_zones'].iterrows():

        dt_when_to_cancel_zone = small_price_bars.loc[dt:, 'FINAL'].lt(demand_zone.LOW)
        dt_when_to_cancel_zone = small_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        demand_zone = pd.DataFrame(dict(zip(demand_zone.index, demand_zone[['HIGH', 'LOW']].values)), index=small_price_bars.loc[dt:dt_when_to_cancel_zone].index).reindex_like(
            small_price_bars[['HIGH', 'LOW']]
        )

        if len(demand_zone) > 0:
            apds.append(
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

    supply_zones_dts = pd.Series([x[i] for x in when_price_hit_which_supply_zone.values for i in range(len(x))]).drop_duplicates()
    for dt, supply_zone in orion_trades['supply_zones'].iterrows():
        dt_when_to_cancel_zone = small_price_bars.loc[dt:, 'FINAL'].gt(supply_zone.HIGH)
        dt_when_to_cancel_zone = small_price_bars.index[
            -1] if not dt_when_to_cancel_zone.any() else dt_when_to_cancel_zone.idxmax()

        supply_zone = pd.DataFrame(dict(zip(supply_zone.index, supply_zone[['HIGH', 'LOW']].values)), index=small_price_bars.loc[dt:dt_when_to_cancel_zone].index).reindex_like(
            small_price_bars[['HIGH', 'LOW']]
        )

        if len(supply_zone) > 0:
            apds.append(
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
        small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
        type='candle',
        show_nontrading=False,
        addplot=apds,
    )

    orion_trades_df = pd.DataFrame(orion_trades)

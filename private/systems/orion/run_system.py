from sysdata.config.configdata import Config

from sysdata.sim.db_futures_sim_data import dbFuturesSimData

from private.systems.orion.rawdata.rawdata import OrionRawData

from private.systems.orion.rules.orion import orion
from private.systems.orion.forecasting import OrionRules

from private.systems.orion.stoplossprofittarget.pathdependency import StopLossProfitTarget

from private.systems.orion.positionsizing.positionsizing import OrionPositionSizing

from private.systems.orion.portfolio.portfolio import OrionPortfolios

from private.systems.orion.accounts.order_simulator.minute_limit_orders import AccountWithOrderSimulatorForLimitOrders

from systems.stage import System


if __name__ == "__main__":

    orion_system = System(
        stage_list=[
            OrionRawData(),
            OrionRules(),
            StopLossProfitTarget(),
            OrionPositionSizing(),
            OrionPortfolios(),
            AccountWithOrderSimulatorForLimitOrders()
        ],
        data=dbFuturesSimData(),
        config=Config('private.systems.orion.orion_config.yaml'),
    )

    orion_portfolio = orion_system.accounts.portfolio()

    import matplotlib.pyplot as plt

    orion_portfolio.percent.curve().plot()
    plt.show()

    subsystem_position = orion_system.positionSize.get_subsystem_position('CL')

    forecast_dict = orion_system.rules.get_raw_forecast('CL', 'orion')

    forecast_after_slpt_dict = orion_system.pathdependency.get_signals_after_limit_price_is_hit_stop_loss_and_profit_target('CL')

    plt.figure()
    forecast_after_slpt_dict['forecasts'].plot()
    plt.show()

    import numpy as np
    import mplfinance as mpf

    small_price_bars = orion_system.rawdata.get_aggregated_minute_prices(
        'CL', barsize=orion_system.config.trading_rules['orion']['other_args']['small_timeframe']
    )
    big_price_bars = orion_system.rawdata.get_aggregated_minute_prices(
        'CL', barsize=orion_system.config.trading_rules['orion']['other_args']['big_timeframe']
    )

    orion_trades = forecast_dict.copy()
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

    new_orion_trades = forecast_after_slpt_dict.copy()
    new_signals = new_orion_trades['forecasts']

    where_values = new_orion_trades['long_limit_prices_after_slpt'].add(
        new_orion_trades['short_limit_prices_after_slpt'], fill_value=0)
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
        small_price_bars.rename(columns=dict(OPEN="Open", HIGH="High", LOW="Low", FINAL="Close")),
        type='candle',
        show_nontrading=False,
        addplot=new_apds,
    )

    from private.systems.orion.stoplossprofittarget.pathdependency import (
        get_signals_after_limit_price_is_hit,
        apply_stop_loss_and_profit_target_to_signals,
    )

    signals_after_limit_price_is_hit_dict = get_signals_after_limit_price_is_hit(
        prices=small_price_bars,
        long_limit_prices=orion_trades['long_limit_prices'],
        short_limit_prices=orion_trades['short_limit_prices'],
        signals=orion_trades['signals'],
        long_zones=orion_trades['long_zones'],
        short_zones=orion_trades['short_zones'],
        long_stop_loss_levels=orion_trades['long_stop_loss_prices'],
        short_stop_loss_levels=orion_trades['short_stop_loss_prices'],
        long_profit_target_levels=orion_trades['long_profit_taker'],
        short_profit_target_levels=orion_trades['short_profit_taker'],
    )

    signals_after_slpt_dict = apply_stop_loss_and_profit_target_to_signals(
        prices=small_price_bars,
        signals=signals_after_limit_price_is_hit_dict['signals'],
        long_stop_loss_levels=signals_after_limit_price_is_hit_dict['new_long_stop_loss_levels'],
        short_stop_loss_levels=signals_after_limit_price_is_hit_dict['new_short_stop_loss_levels'],
        long_profit_target_levels=signals_after_limit_price_is_hit_dict['new_long_profit_target_levels'],
        short_profit_target_levels=signals_after_limit_price_is_hit_dict['new_short_profit_target_levels'],
        long_limit_prices=signals_after_limit_price_is_hit_dict['new_long_limit_prices'],
        short_limit_prices=signals_after_limit_price_is_hit_dict['new_short_limit_prices'],
    )




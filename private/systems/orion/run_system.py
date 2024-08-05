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

    plt.figure()
    orion_portfolio.curve().plot()
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
    long_signals = orion_trades['long_signals']
    short_signals = orion_trades['short_signals']

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

    import pandas as pd

    path_dep_df = forecast_after_slpt_dict.pop('path_dep_df')

    new_orion_trades = pd.DataFrame(forecast_after_slpt_dict).tz_convert('EST')
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

    trades = pd.DataFrame(dict(signal=np.nan, entry_price=np.nan, exit_price=np.nan), index=small_price_bars.index)
    for signal_dt, trade_metadata in path_dep_df.iterrows():

        entry_dt = trade_metadata.dt_when_limit_price_was_hit
        exit_dt = trades.loc[trade_metadata.dt_when_trade_exited:].index[1]
        idxs_in_the_market = trades.loc[entry_dt:exit_dt].index

        trades.loc[idxs_in_the_market, 'signal'] = trade_metadata.signals
        trades.loc[idxs_in_the_market, 'entry_price'] = small_price_bars.shift(1).loc[idxs_in_the_market, 'FINAL']
        trades.loc[idxs_in_the_market, 'exit_price'] = small_price_bars.loc[idxs_in_the_market, 'OPEN']

        trades.loc[entry_dt, 'entry_price'] = new_orion_trades.loc[entry_dt, 'long_limit_prices_after_slpt']
        if trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_session_ended:
            trades.loc[exit_dt, 'exit_price'] = small_price_bars.FINAL.asof(trade_metadata.dt_when_trade_exited)
        elif trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_stop_loss_was_hit:
            trades.loc[exit_dt, 'exit_price'] = new_orion_trades.loc[trade_metadata.dt_when_trade_exited, 'stop_loss_levels_after_slpt']
        elif trade_metadata.dt_when_trade_exited == trade_metadata.dt_when_profit_target_was_hit:
            trades.loc[exit_dt, 'exit_price'] = new_orion_trades.loc[trade_metadata.dt_when_trade_exited, 'profit_target_levels_after_slpt']
        else:
            input("WARNING! Continue?")


    # from syscore.fileutils import resolve_path_and_filename_for_package
    #
    # path_dep_df_summary = path_dep_df[
    #     ['signals', 'dt_when_limit_price_was_hit', 'dt_when_stop_loss_was_hit', 'dt_when_profit_target_was_hit', 'dt_when_this_session_ended', 'dt_when_trade_exited']
    # ].tz_convert('EST')
    #
    # path_dep_df_summary['dt_when_limit_price_was_hit'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_limit_price_was_hit']]
    # # path_dep_df_summary['dt_when_zone_changed'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_zone_changed']]
    # # path_dep_df_summary['dt_when_stop_loss_was_hit'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_stop_loss_was_hit']]
    # path_dep_df_summary['dt_when_profit_target_was_hit'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_profit_target_was_hit']]
    # path_dep_df_summary['dt_when_this_session_ended'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_this_session_ended']]
    # path_dep_df_summary['dt_when_trade_exited'] = [x.tz_convert('EST') for x in path_dep_df_summary['dt_when_trade_exited']]
    #
    # path_dep_df_summary.to_csv(resolve_path_and_filename_for_package('private.systems.orion.path_dep.csv'), sep='\t')
    #
    # order_simulator = orion_system.accounts.get_order_simulator('CL', is_subsystem=False)
    #
    # orion_trades_df = pd.DataFrame({k: orion_trades[k] for k in orion_trades if k not in ['long_zones', 'short_zones']}).tz_convert('EST')
    #
    # diagnostic_df = order_simulator.diagnostic_df()

    #########################################################################################################################

    # import pandas as pd
    #
    # price_bars = orion_system.rawdata.get_minute_prices('CL')
    # sessions = orion_system.data.get_sessions_for_instrument('CL')
    #
    # orion_rules_result = orion(price_bars, sessions, rr=2.5)
    #
    # orion_rules_result_df = pd.DataFrame({k: orion_rules_result[k] for k in orion_rules_result if k not in ['long_zones', 'short_zones']})


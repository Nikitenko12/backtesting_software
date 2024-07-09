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


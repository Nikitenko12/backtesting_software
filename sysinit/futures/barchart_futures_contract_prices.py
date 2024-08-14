from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
import os
from itertools import compress
from syscore.fileutils import (
    get_resolved_pathname,
    files_with_extension_in_resolved_pathname,
)
from syscore.dateutils import (
    month_from_contract_letter, MIXED_FREQ, HOURLY_FREQ, DAILY_PRICE_FREQ, MINUTE_FREQ,
)

from sysinit.futures.contract_prices_from_csv_to_arctic import (
    init_db_with_csv_futures_contract_prices,
)


def strip_file_names(pathname):
    # These won't have .csv attached
    resolved_pathname = get_resolved_pathname(pathname)
    file_names = files_with_extension_in_resolved_pathname(resolved_pathname)
    frequency_dict = {"Day": DAILY_PRICE_FREQ, "Hour": HOURLY_FREQ, "Minute": MINUTE_FREQ}
    for filename in file_names:
        freq_ident_contract = filename.split("_")
        frequency = frequency_dict[freq_ident_contract[0]]
        identifier = freq_ident_contract[1]
        datecode = freq_ident_contract[2]

        instrument = market_map[identifier]

        new_path_name = os.path.join(resolved_pathname, f'{freq_ident_contract[0]}')
        try:
            os.mkdir(new_path_name)
        except:
            print("Directory %s already exists" % new_path_name)
        new_file_name = "%s_%s.csv" % (instrument, datecode)
        new_full_name = os.path.join(new_path_name, new_file_name)
        old_full_name = os.path.join(resolved_pathname, filename + ".csv")

        print("Rename %s to\n %s" % (old_full_name, new_full_name))

        os.rename(old_full_name, new_full_name)

    return frequency_dict


# market_map = dict(
#     AE="AEX",
#     A6="AUD",
#     HR="BOBL",
#     II="BTP",
#     GG="BUND",
#     MX="CAC",
#     BJ="CHEESE",
#     KG="COTTON",
#     HG="COPPER",
#     ZC="CORN",
#     CL="CRUDE_W",
#     GE="EDOLLAR",
#     E6="EUR",
#     NG="GAS_US",
#     B6="GBP",
#     GC="GOLD",
#     J6="JPY",
#     HE="LEANHOG",
#     LE="LIVECOW",
#     DF="MILKDRY",
#     DK="MILKWET",
#     M6="MXP",
#     NQ="NASDAQ",
#     N6="NZD",
#     FN="OAT",
#     PA="PALLAD",
#     HF="SHATZ",
#     PL="PLAT",
#     ZS="SOYBEAN",
#     ES="SP500",
#     ZT="US2",
#     ZF="US5",
#     ZN="US10",
#     ZB="US20",
#     VI="VIX",
#     ZW="WHEAT",
#     DV="V2X",
#     UD="US30",
#     FX="EUROSTX",
#     GR="GOLD_micro",
#     NM="NASDAQ_micro",
#     QM="CRUDE_W_mini",
#     QG="GAS_US_mini",
#     ET="SP500_micro",
# )
market_map = dict(
    NQ="NQ",
    ES="ES",
    GC="GC",
    CL="CL",
)

barchart_csv_config = ConfigCsvFuturesPrices(
    input_date_index_name="Time",
    input_skiprows=0,
    input_skipfooter=1,
    input_date_format="%Y-%m-%dT%H:%M:%S",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)


def transfer_barchart_prices_to_arctic(datapath, frequencies):
    # frequencies = strip_file_names(datapath)
    # for frequency in frequencies.items():
    #     init_db_with_csv_futures_contract_prices(
    #         '.'.join([datapath, frequency[0]]), csv_config=barchart_csv_config, frequency=frequency[1]
    #     )
    for frequency in frequencies:
        init_db_with_csv_futures_contract_prices(
            datapath, csv_config=barchart_csv_config, frequency=frequency
        )


if __name__ == "__main__":
    input("Will overwrite existing prices are you sure?! CTL-C to abort")
    # modify flags as required
    datapath = "sysinit.futures.bcutils.data.CL"
    transfer_barchart_prices_to_arctic(datapath, [MINUTE_FREQ])

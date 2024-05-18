"""
Read and write data from mongodb for individual futures contracts

"""

from syscore.dateutils import Frequency, MIXED_FREQ

from sysdata.influxdb.influx_connection import influxData
from sysobjects.contracts import listOfFuturesContracts
from sysdata.futures.futures_per_contract_prices import (
    futuresContractPriceData,
)
from sysobjects.futures_per_contract_prices import futuresContractPrices
from sysobjects.contracts import futuresContract, get_code_and_id_from_contract_key
from syslogging.logger import *

import pandas as pd

CONTRACT_BUCKET = "futures_contract_prices"


class influxFuturesContractPriceData(futuresContractPriceData):
    """
    Class to read / write futures price data to and from influx
    """

    def __init__(self, influx_db=None, log=get_logger("influxFuturesContractPriceData")):
        super().__init__(log=log)

        self._influx_connection = influxData(CONTRACT_BUCKET, influx_db=influx_db)

    def __repr__(self):
        return repr(self._influx_connection)

    @property
    def influx_connection(self):
        return self._influx_connection

    def _get_merged_prices_for_contract_object_no_checking(
        self, futures_contract_object: futuresContract
    ) -> futuresContractPrices:
        """
        Read back the prices for a given contract object

        :param futures_contract_object:  futuresContract
        :return: data
        """

        # Returns a data frame which should have the right format
        data = self._get_prices_at_frequency_for_contract_object_no_checking(
            futures_contract_object, frequency=MIXED_FREQ
        )

        return data

    def _get_prices_at_frequency_for_contract_object_no_checking(
        self, futures_contract_object: futuresContract, frequency: Frequency
    ) -> futuresContractPrices:
        ident = from_contract_and_freq_to_key(
            futures_contract_object, frequency=frequency
        )

        # Returns a data frame which should have the right format
        data = self.influx_connection.read(ident[0], tag=("frequency", ident[1]))

        return futuresContractPrices(data)

    def _write_merged_prices_for_contract_object_no_checking(
        self,
        futures_contract_object: futuresContract,
        futures_price_data: futuresContractPrices,
    ):
        """
        Write prices
        CHECK prices are overridden on second write

        :param futures_contract_object: futuresContract
        :param futures_price_data: futuresContractPriceData
        :return: None
        """

        self._write_prices_at_frequency_for_contract_object_no_checking(
            futures_contract_object=futures_contract_object,
            frequency=MIXED_FREQ,
            futures_price_data=futures_price_data,
        )

    def _write_prices_at_frequency_for_contract_object_no_checking(
        self,
        futures_contract_object: futuresContract,
        futures_price_data: futuresContractPrices,
        frequency: Frequency,
    ):
        ident = from_contract_and_freq_to_key(
            futures_contract_object, frequency=frequency
        )
        futures_price_data_as_pd = pd.DataFrame(futures_price_data)

        self.influx_connection.write(ident[0], futures_price_data_as_pd, tags={"frequency": ident[1]})

        self.log.debug(
            "Wrote %s lines of prices for %s at %s to %s"
            % (
                len(futures_price_data),
                str(futures_contract_object.key),
                str(frequency),
                str(self),
            ),
            **futures_contract_object.log_attributes(),
            method="temp",
        )

    def get_contracts_with_merged_price_data(self) -> listOfFuturesContracts:
        """

        :return: list of contracts
        """

        list_of_contracts = self.get_contracts_with_price_data_for_frequency(
            frequency=MIXED_FREQ
        )

        return list_of_contracts

    def get_contracts_with_price_data_for_frequency(
        self, frequency: Frequency
    ) -> listOfFuturesContracts:
        list_of_contract_and_freq_tuples = (
            self._get_contract_and_frequencies_with_price_data()
        )
        list_of_contracts = [
            freq_and_contract_tuple[1]
            for freq_and_contract_tuple in list_of_contract_and_freq_tuples
            if freq_and_contract_tuple[0] == frequency
        ]

        list_of_contracts = listOfFuturesContracts(list_of_contracts)

        return list_of_contracts

    def has_merged_price_data_for_contract(
        self, contract_object: futuresContract
    ) -> bool:
        return self.has_price_data_for_contract_at_frequency(
            contract_object, frequency=MIXED_FREQ
        )

    def has_price_data_for_contract_at_frequency(
        self, contract_object: futuresContract, frequency: Frequency
    ) -> bool:
        ident = from_contract_and_freq_to_key(
            contract_object, frequency=frequency
        )

        return self.influx_connection.has_keyname_and_tag(ident[0], ident[1])

    def _get_contract_and_frequencies_with_price_data(self) -> list:
        """

        :return: list of futures contracts as tuples
        """

        all_keynames_and_tags = self._all_keynames_and_tags_in_library()
        list_of_contract_and_freq_tuples = [
            from_key_and_tag_to_freq_and_contract(keyname, tag)
            for keyname in all_keynames_and_tags.keys()
            for tag in all_keynames_and_tags[keyname]
        ]

        return list_of_contract_and_freq_tuples

    def _all_keynames_and_tags_in_library(self) -> dict:
        return self.influx_connection.get_keynames_and_tags()

    def _delete_merged_prices_for_contract_object_with_no_checks_be_careful(
        self, futures_contract_object: futuresContract
    ):
        """
        Delete prices for a given contract object without performing any checks

        WILL THIS WORK IF DOESN'T EXIST?
        :param futures_contract_object:
        :return: None
        """

        self._delete_prices_at_frequency_for_contract_object_with_no_checks_be_careful(
            futures_contract_object, frequency=MIXED_FREQ
        )

    def _delete_prices_at_frequency_for_contract_object_with_no_checks_be_careful(
        self, futures_contract_object: futuresContract, frequency: Frequency
    ):
        ident = from_contract_and_freq_to_key(
            contract=futures_contract_object, frequency=frequency
        )
        self.influx_connection.delete(ident[0], ("frequency", ident[1]))
        self.log.debug(
            "Deleted all prices for %s from %s"
            % (futures_contract_object.key, str(self)),
            **futures_contract_object.log_attributes(),
            method="temp",
        )


def from_key_and_tag_to_freq_and_contract(keyname, tag):
    contract_str = keyname
    if len(tag) == 1:
        frequency = MIXED_FREQ
    else:
        frequency = Frequency[tag]
    contract_str_split = contract_str.split(".")
    futures_contract = futuresContract(contract_str_split[0], contract_str_split[1])

    return frequency, futures_contract


def from_contract_and_freq_to_key(contract: futuresContract, frequency: Frequency):
    if frequency is MIXED_FREQ:
        frequency_str = ""
    else:
        frequency_str = frequency.name + "/"

    return from_tuple_to_key(
        [frequency_str, contract.instrument_code, contract.date_str]
    )


def from_tuple_to_key(keytuple):
    return keytuple[1] + "." + keytuple[2], keytuple[0]

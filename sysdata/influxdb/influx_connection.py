import pandas as pd
from influxdb_client import InfluxDBClient

from syscore.constants import arg_not_supplied
from sysdata.config.production_config import get_production_config

"""


"""
LIST_OF_INFLUX_PARAMS = ["influx_url", "influx_token", "influx_org"]


def influx_defaults(**kwargs):
    """
    Returns influx configuration with following precedence

    1- if passed in arguments: influx_url, influx_token, influx_org - use that
    2- if defined in private_config file, use that. influx_url, influx_token, influx_org
    3- if defined in system defaults file, use that: influx_url, influx_token, influx_org

    :return: influx_url, influx_token, influx_org
    """
    # this will include defaults.yaml if not defined in private
    passed_param_names = list(kwargs.keys())
    production_config = get_production_config()
    output_dict = {}
    for param_name in LIST_OF_INFLUX_PARAMS:
        if param_name in passed_param_names:
            param_value = kwargs[param_name]
        else:
            param_value = arg_not_supplied

        if param_value is arg_not_supplied:
            param_value = getattr(production_config, param_name)

        output_dict[param_name] = param_value

    # Get from dictionary
    influx_url = output_dict["influx_url"]
    token = output_dict["influx_token"]
    org = output_dict["influx_org"]

    return influx_url, token, org


class InfluxClientFactory(object):
    """
    Only one InfluxDBClient is needed per Python process and InfluxDB instance.

    I'm not sure why anyone would need more than one InfluxDB instance,
    but it's easy to support, so why not?
    """

    def __init__(self):
        self.influx_clients = {}

    def get_influx_client(self, url, token, org):
        key = (url, token, org)
        if key in self.influx_clients:
            return self.influx_clients.get(key)
        else:
            client = InfluxDBClient(url, token, org)
            self.influx_clients[key] = client
            return client


# Only need one of these
influx_client_factory = InfluxClientFactory()


class influxData(object):
    """
    All of our Influx connections use this class

    """
    def __init__(
        self,
        influx_bucket_name,
        influx_url: str = arg_not_supplied,
        influx_token: str = arg_not_supplied,
        influx_org: str = arg_not_supplied,
    ):
        url, token, org = influx_defaults(
            influx_url=influx_url,
            influx_token=influx_token,
            influx_org=influx_org,
        )

        self.url = url
        self.token = token
        self.org = org

        client = influx_client_factory.get_influx_client(url, token, org)

        self.client = client
        self.bucket_name = influx_bucket_name
        self.bucket = self._setup_bucket(client, self.bucket_name)

        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()

    def __repr__(self):
        return (
            f"InfluxDB connection: url {self.url}, "
            f"bucket {self.bucket_name}"
        )

    def read(self, ident) -> pd.DataFrame:
        query = f'from(bucket:"{self.bucket_name}")' \
            f' |> range(start: 0, stop: now())' \
            f' |> filter(fn: (r) => r._measurement == "{ident}")' \
            f' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        item = self.query_api.query_data_frame(query)
        return item

    def write(self, ident: str, data: pd.DataFrame, tags: dict):
        df = data.copy()
        for tag in list(tags.keys()):
            df[tag] = tags[tag]

        self.write_api.write(
            bucket=self.bucket,
            record=df,
            data_frame_measurement_name=ident,
            data_frame_tag_columns=[list(tags.keys())],
        )

    def get_keynames(self) -> list:
        query = f'import \"influxdata/influxdb/schema\"' \
            f'schema.measurements(bucket: \"{self.bucket_name}\")'
        keynames = self.query_api.query(query)

        return keynames

    def has_keyname(self, keyname) -> bool:
        all_keynames = self.get_keynames()

        return keyname in all_keynames

    def delete(self, ident: str):
        delete_api = self.client.delete_api()
        delete_api.delete(predicate=f'_measurement="{ident}"')

    def _setup_bucket(self, client: InfluxDBClient, bucket_name):
        buckets_api = client.buckets_api()
        if buckets_api.find_bucket_by_name(bucket_name) is None:
            buckets_api.create_bucket(bucket_name=bucket_name)

        return buckets_api.find_bucket_by_name(bucket_name)


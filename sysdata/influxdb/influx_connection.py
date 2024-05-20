import datetime

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

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


class influxDb:
    """
    Keeps track of influx database we are connected to

    But requires adding a collection with influxData before useful
    """
    def __init__(
        self,
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


class influxData(object):
    """
    All of our Influx connections use this class

    """
    def __init__(
        self,
        influx_bucket_name: str,
        influx_db: influxDb = None
    ):
        if influx_db is None:
            influx_db = influxDb()

        self.url = influx_db.url
        self.org = influx_db.org
        self.client = influx_db.client

        self.bucket_name = influx_bucket_name
        self.bucket = self._setup_bucket(self.bucket_name)

        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def __repr__(self):
        return (
            f"InfluxDB connection: url {self.url}, "
            f"bucket {self.bucket_name}"
        )

    def read(self, ident, tag: tuple = None) -> pd.DataFrame:
        if tag is not None:
            query = f'from(bucket:"{self.bucket_name}")' \
                    f' |> range(start: 0, stop: now())' \
                    f' |> filter(fn: (r) => r._measurement == "{ident}")' \
                    f' |> filter(fn: (r) => r.{tag[0]} == "{tag[1]}")' \
                    f' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        else:
            query = f'from(bucket:"{self.bucket_name}")' \
                f' |> range(start: 0, stop: now())' \
                f' |> filter(fn: (r) => r._measurement == "{ident}")' \
                f' |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

        item = self.query_api.query_data_frame(query, org=self.org)
        if isinstance(item, list):
            item = item[0]

        return item

    def write(self, ident: str, data: pd.DataFrame, tags: dict = None):
        df = data.copy()
        for tag in list(tags.keys()):
            df[tag] = tags[tag]

        self.write_api.write(
            bucket=self.bucket_name,
            org=self.org,
            record=df,
            data_frame_measurement_name=ident,
            data_frame_tag_columns=list(tags.keys()),
        )

    def get_keynames(self) -> list:
        query = f'''import \"influxdata/influxdb/schema\"
        
                schema.measurements(bucket: \"{self.bucket_name}\")'''
        keynames = self.query_api.query(query, org=self.org)[0].records

        return keynames

    def has_keyname(self, keyname) -> bool:
        all_keynames = self.get_keynames()

        return keyname in all_keynames

    def get_keynames_and_tags(self) -> dict:
        measurements = self.get_keynames()
        measurements = [x.values['_value'] for x in measurements]
        keynames_and_tags = dict()

        for measurement in measurements:
            query = f'''import \"influxdata/influxdb/schema\"

                    schema.measurementTagValues(bucket: "{self.bucket_name}", tag: "frequency", measurement: "{measurement}")'''

            these_tags = self.query_api.query(query, org=self.org)[0].records
            keynames_and_tags[measurement] = [x.values['_value'] for x in these_tags]

        return keynames_and_tags

    def has_keyname_and_tag(self, keyname, tag) -> bool:
        keynames_and_tags = self.get_keynames_and_tags()
        return tag in keynames_and_tags[keyname]

    def delete(self, ident: str, tag: tuple = None):
        predicate = f'_measurement="{ident}"'
        if tag is not None:
            predicate += f', {tag[0]}="{tag[1]}"'
        delete_api = self.client.delete_api()
        delete_api.delete(
            bucket=self.bucket_name,
            start=datetime.datetime.fromordinal(0),
            stop=datetime.datetime.now(),
            predicate=predicate
        )

    def _setup_bucket(self, bucket_name):
        if bucket_name is arg_not_supplied:
            return None

        buckets_api = self.client.buckets_api()
        if buckets_api.find_bucket_by_name(bucket_name) is None:
            buckets_api.create_bucket(bucket_name=bucket_name, org_id=self.org)

        return buckets_api.find_bucket_by_name(bucket_name)


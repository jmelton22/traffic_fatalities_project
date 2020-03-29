#!/usr/bin/env python3

import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


def main():
    path = '../data/states'
    bq_assistant = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')

    """
    Per State Overview
        - Fatal accidents per state
        - Accident Fatalities per state
        - Fatalities / Accident per state
    """

    QUERY = """
        SELECT
            state_name,
            COUNT(consecutive_number) AS accidents,
            SUM(number_of_fatalities) AS fatalities,
            SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
        GROUP BY state_name
        ORDER BY fatalities_per_accident DESC
    """

    accident_fatalities_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))
    accident_fatalities_2015['year'] = 2015
    accident_fatalities_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))
    accident_fatalities_2016['year'] = 2016

    accident_fatalities_2015.to_csv(f'{path}/accident_fatalities_2015.csv', index=False)
    accident_fatalities_2016.to_csv(f'{path}/accident_fatalities_2016.csv', index=False)


if __name__ == '__main__':
    main()

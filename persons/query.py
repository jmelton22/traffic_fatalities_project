#!/usr/bin/env python3

import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


def main():
    path = '../data/persons'
    bq_assistant = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')

    QUERY = """
        SELECT
            p.state_number,
            p.consecutive_number,
            p.vehicle_number,
            p.person_number,
            person_type_name AS person_type,
            age,
            sex,
            injury_severity_name AS injury_severity,
            land_use_name AS land_use,
            functional_system_name AS trafficway_type,
            manner_of_collision_name AS manner_of_collision,
            body_type_name AS body_type,
            rollover,
            seating_position_name AS seating_position,
            ejection_name AS ejection,
            restraint_system_helmet_use_name AS safety_equipment_use,
            air_bag_deployed_name AS air_bag_deployed,
            non_motorist_safety_equipment_use
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.person_{0}` AS p
        LEFT JOIN
            `bigquery-public-data.nhtsa_traffic_fatalities.safetyeq_{0}` AS s
            ON p.consecutive_number=s.consecutive_number AND p.vehicle_number=s.vehicle_number AND p.person_number=s.person_number
        ORDER BY
            consecutive_number, vehicle_number, person_number
    """

    data_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))
    data_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))

    data_2015.to_csv(f'{path}/person_data_2015.csv', index=False)
    data_2016.to_csv(f'{path}/person_data_2016.csv', index=False)


if __name__ == '__main__':
    main()

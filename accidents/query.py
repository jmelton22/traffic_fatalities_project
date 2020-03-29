#!/usr/bin/env python3

import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


def main():
    path = '~/data/accidents'
    bq_assistant = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')

    QUERY = """
        SELECT
            a.consecutive_number,
            veh.vehicle_number,
            state_name,
            a.hour_of_crash AS hour_of_day,
            CASE
                WHEN day_of_week = 1 THEN 'Sunday'
                WHEN day_of_week = 2 THEN 'Monday'
                WHEN day_of_week = 3 THEN 'Tuesday'
                WHEN day_of_week = 4 THEN 'Wednesday'
                WHEN day_of_week = 5 THEN 'Thursday'
                WHEN day_of_week = 6 THEN 'Friday'
                WHEN day_of_week = 7 THEN 'Saturday'
            END day_of_week,
            CASE
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 1 THEN 'January'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 2 THEN 'February'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 3 THEN 'March'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 4 THEN 'April'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 5 THEN 'May'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 6 THEN 'June'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 7 THEN 'July'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 8 THEN 'August'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 9 THEN 'September'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 10 THEN 'October'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 11 THEN 'November'
                WHEN EXTRACT(MONTH FROM a.timestamp_of_crash) = 12 THEN 'December'
            END month,
            EXTRACT(YEAR FROM a.timestamp_of_crash) AS year,
            land_use_name AS land_use,
            national_highway_system,
            route_signing_name AS roadway_type,
            type_of_intersection AS intersection,
            light_condition_name AS light_condition,
            atmospheric_conditions_1_name AS atmospheric_conditions,
            latitude,
            longitude,
            manner_of_collision_name AS manner_of_collision,
            number_of_vehicle_forms_submitted_all AS num_vehicles,
            number_of_persons_not_in_motor_vehicles_in_transport_mvit AS num_nonmotorists,
            number_of_persons_in_motor_vehicles_in_transport_mvit AS num_motorists,
            number_of_fatalities AS num_fatalities,
            number_of_drunk_drivers AS num_drunk_drivers,
            body_type_name AS body_type,
            vehicle_model_year AS vehicle_year,
            contributing_circumstances_motor_vehicle_name AS vehicle_conditions,
            previous_dwi_convictions,
            previous_speeding_convictions,
            speeding_related,
            speed_limit,
            drivers_vision_obscured_by_name AS driver_vision_obscured,
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}` AS a
        JOIN
            `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}` AS veh
            ON a.consecutive_number=veh.consecutive_number
        JOIN
            `bigquery-public-data.nhtsa_traffic_fatalities.factor_{0}` AS f
            ON a.consecutive_number=f.consecutive_number AND veh.vehicle_number=f.vehicle_number
        JOIN
            `bigquery-public-data.nhtsa_traffic_fatalities.vision_{0}` AS v
            ON a.consecutive_number=v.consecutive_number AND veh.vehicle_number=v.vehicle_number
        ORDER BY consecutive_number, vehicle_number
    """

    data_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))
    data_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))

    data_2015.to_csv(f'{path}/accident_driver_data_2015.csv', index=False)
    data_2016.to_csv(f'{path}/accident_driver_data_2016.csv', index=False)


if __name__ == '__main__':
    main()

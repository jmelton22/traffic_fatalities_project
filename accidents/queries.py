#!/usr/bin/env python3

import pandas as pd
import numpy as np
from google.cloud import bigquery
from bq_helper import BigQueryHelper


def main():
    path = '~/data/accidents'
    bq_assistant = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')
    """
    Information about individual accidents
    """

    QUERY = """
        SELECT
            consecutive_number,
            state_name,
            hour_of_crash AS hour_of_day,
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
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 1 THEN 'January'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 2 THEN 'February'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 3 THEN 'March'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 4 THEN 'April'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 5 THEN 'May'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 6 THEN 'June'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 7 THEN 'July'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 8 THEN 'August'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 9 THEN 'September'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 10 THEN 'October'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 11 THEN 'November'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 12 THEN 'December'
            END month,
            EXTRACT(YEAR FROM timestamp_of_crash) AS year,
            land_use_name AS land_use,
            national_highway_system,
            functional_system_name AS trafficway_type,
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
            number_of_drunk_drivers AS num_drunk_drivers
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
        ORDER BY consecutive_number
    """

    accidents_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))
    accidents_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))

    accidents_2015 = accident_data_prep(accidents_2015)
    accidents_2016 = accident_data_prep(accidents_2016)

    accidents = pd.concat([accidents_2015, accidents_2016]).reset_index(drop=True)
    accidents.info()

    accidents.to_csv(f'{path}/accident_data.csv', index=False)


def accident_data_prep(accident_data):
    # Replace Unknown and Not Reported values with NaN
    accident_data.replace({'Unknown': np.nan,
                           'Not Reported': np.nan,
                           'Trafficway Not in State Inventory': np.nan},
                          inplace=True)

    # Replace unknown hour values (hour = 99) with NaN
    accident_data['hour_of_day'].replace({99: np.nan}, inplace=True)

    # Replace unknown national highway system values (Unknown = 9)
    accident_data['national_highway_system'].replace({9: np.nan}, inplace=True)

    # Replace bad latitude/longitude values
    accident_data['longitude'].where(accident_data['longitude'] < 0, np.nan, inplace=True)
    accident_data['latitude'].where(accident_data['latitude'] < 70, np.nan, inplace=True)

    # Feature Engineering

    def hour_category(x):
        """
            'night': 10PM - 4AM
            'morning': 4AM - 10AM
            'day': 10AM - 4PM
            'evening': 4PM - 10PM
        """
        if x in [22, 23, 0, 1, 2, 3]:
            return 'night'
        elif x in [4, 5, 6, 7, 8, 9]:
            return 'morning'
        elif x in [10, 11, 12, 13, 14, 15]:
            return 'day'
        elif x in [16, 17, 18, 19, 20, 21]:
            return 'evening'
        else:
            return np.nan

    # Create part of day categorical column
    accident_data['part_of_day'] = accident_data['hour_of_day'].apply(hour_category)

    # Create binary class column: is_weekend (0: weekday, 1: weekend)
    accident_data['is_weekend'] = accident_data['day_of_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Create binary class column: multiple_vehicles (0: single vehicle; 1: multiple vehicles)
    accident_data['multiple_vehicles'] = accident_data['num_vehicles'].apply(lambda x: 1 if x > 1 else 0)

    # Create binary class column: nonmotorist_involed (0: no non-motorists; 1: non-motorist(s) involved)
    accident_data['nonmotorist_involved'] = accident_data['num_nonmotorists'].apply(lambda x: 1 if x > 0 else 0)

    # Create binary class column: multiple_motorists (0: single motorist; 1: multiple motorists)
    accident_data['multiple_motorists'] = accident_data['num_motorists'].apply(lambda x: 1 if x > 1 else 0)

    # Create binary class column: drunk_driver_involved (0: no drunk driver; 1: drunk driver involved)
    accident_data['drunk_driver_involved'] = accident_data['num_drunk_drivers'].apply(lambda x: 1 if x > 0 else 0)

    # Create binary class label column: multiple fatalities (0: single fatality accident; 1: multiple fatalities)
    accident_data['multiple_fatalities'] = accident_data['num_fatalities'].apply(lambda x: 1 if x > 1 else 0)

    # Drop NaNs and clean up
    accident_data.dropna(inplace=True)
    accident_data['hour_of_day'] = accident_data['hour_of_day'].astype(int)
    accident_data['national_highway_system'] = accident_data['national_highway_system'].astype(int)

    accident_data.reset_index(drop=True, inplace=True)

    return accident_data


if __name__ == '__main__':
    main()

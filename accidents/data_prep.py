#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.utils import resample


def main():
    path = '../data/accidents'
    data_2015 = pd.read_csv(f'{path}/accident_driver_data_2015.csv', header=0)
    data_2016 = pd.read_csv(f'{path}/accident_driver_data_2016.csv', header=0)

    data_2015 = data_prep(data_2015)
    data_2016 = data_prep(data_2016)

    accident_data = pd.concat([data_2015, data_2016])
    accident_data.reset_index(drop=True, inplace=True)
    accident_data.info()

    accident_data.to_csv(f'{path}/accident_data_clean.csv', index=False)

    df_majority = accident_data[accident_data['multiple_fatalities'] == 0]
    df_minority = accident_data[accident_data['multiple_fatalities'] == 1]

    df_minority_resampled = resample(df_minority,
                                     n_samples=len(df_majority),
                                     replace=True,
                                     random_state=2020)
    balanced_data = pd.concat([df_majority, df_minority_resampled])
    balanced_data.reset_index(drop=True, inplace=True)
    balanced_data.to_csv(f'{path}/accident_data_clean_balanced.csv', index=False)


def data_prep(data):
    # Replace Unknown and Not Reported values with NaN
    data.replace({'Unknown': np.nan,
                  'Not Reported': np.nan,
                  'Trafficway Not in State Inventory': np.nan,
                  'Not Reported (Since 2010)': np.nan,
                  'Unknown Body Type': np.nan},
                 inplace=True)

    # Replace unknown hour values (hour = 99) with NaN
    data['hour_of_day'].replace({99: np.nan}, inplace=True)

    # Replace unknown national highway system values (Unknown = 9)
    data['national_highway_system'].replace({9: np.nan}, inplace=True)

    # Replace bad latitude/longitude values
    data['longitude'].where(data['longitude'] < 0, np.nan, inplace=True)
    data['latitude'].where(data['latitude'] < 70, np.nan, inplace=True)

    data['previous_dwi_convictions'].replace({99: np.nan, 998: np.nan}, inplace=True)
    data['previous_speeding_convictions'].replace({99: np.nan, 998: np.nan}, inplace=True)
    data['speed_limit'].replace({98: np.nan, 99: np.nan}, inplace=True)
    data['vehicle_year'].replace({9998: np.nan, 9999: np.nan}, inplace=True)

    data['body_type'].replace(r'Unknown', np.nan, regex=True, inplace=True)

    data.dropna(inplace=True)

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

    def binarize_col(x):
        if x > 0:
            return 1
        else:
            return 0

    def speeding_binary(x):
        if 'Yes' in x:
            return 1
        elif 'No' in x:
            return 0

    def vision_binary(x):
        if x == 'No Obstruction Noted' or x == 'No Driver Present/Unknown if Driver Present':
            return 0
        else:
            return 1

    # Refactor land_use column to binary (0: Urban; 1: Rural)
    data['land_use'].replace({'Urban': 1,
                              'Rural': 0},
                             inplace=True)
    data.rename(columns={'land_use': 'land_use_urban'}, inplace=True)

    # Create part of day categorical column
    data['part_of_day'] = data['hour_of_day'].apply(hour_category)

    # Create binary class column: is_weekend (0: weekday, 1: weekend)
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Create binary class column: multiple_vehicles (0: single vehicle; 1: multiple vehicles)
    data['multiple_vehicles'] = data['num_vehicles'].apply(lambda x: 1 if x > 1 else 0)

    # Create binary class column: nonmotorist_involed (0: no non-motorists; 1: non-motorist(s) involved)
    data['nonmotorist_involved'] = data['num_nonmotorists'].apply(lambda x: 1 if x > 0 else 0)

    # Create binary class column: multiple_motorists (0: single motorist; 1: multiple motorists)
    data['multiple_motorists'] = data['num_motorists'].apply(lambda x: 1 if x > 1 else 0)

    # Create binary class column: drunk_driver_involved (0: no drunk driver; 1: drunk driver involved)
    data['drunk_driver_involved'] = data['num_drunk_drivers'].apply(lambda x: 1 if x > 0 else 0)

    # Create binary class label column: multiple fatalities (0: single fatality accident; 1: multiple fatalities)
    data['multiple_fatalities'] = data['num_fatalities'].apply(lambda x: 1 if x > 1 else 0)

    data['previous_dwi_convictions'] = data['previous_dwi_convictions'].apply(binarize_col)
    data['previous_speeding_convictions'] = data['previous_speeding_convictions'].apply(binarize_col)
    data['speeding_related'] = data['speeding_related'].apply(speeding_binary)
    data['driver_vision_obscured'] = data['driver_vision_obscured'].apply(vision_binary)

    # Reset columns types to integers
    cols = ['hour_of_day', 'national_highway_system', 'previous_dwi_convictions', 'vehicle_year',
            'previous_speeding_convictions', 'speeding_related', 'speed_limit', 'driver_vision_obscured']
    for col in cols:
        data[col] = data[col].astype(int)

    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    main()

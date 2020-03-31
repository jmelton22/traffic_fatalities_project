#!/usr/bin/env python3

import pandas as pd
import numpy as np


def main():
    path = '../data/persons'
    data_2015 = pd.read_csv(f'{path}/person_data_2015.csv', header=0)
    data_2016 = pd.read_csv(f'{path}/person_data_2016.csv', header=0)

    data_2015 = data_prep(data_2015)
    data_2016 = data_prep(data_2016)

    accident_data = pd.concat([data_2015, data_2016])
    accident_data.reset_index(drop=True, inplace=True)

    accident_data.to_csv(f'{path}/person_data_clean.csv', index=False)


def data_prep(data):

    def combine_safety(x, y):
        if x == 'Not a Motor Vehicle Occupant':
            return y
        else:
            return x

    data['safety_equipment_use'] = data.apply(lambda row: combine_safety(row['safety_equipment_use'], row['non_motorist_safety_equipment_use']), axis=1)
    data.drop('non_motorist_safety_equipment_use', axis=1, inplace=True)

    data.replace({'Unknown': np.nan,
                  'Not Reported': np.nan,
                  'Trafficway Not in State Inventory': np.nan,
                  'Unknown Body Type': np.nan,
                  'Unknown if Ejected (Since 2009)': np.nan,
                  'Unknown if Used': np.nan,
                  'Unknown if Helmet Worn': np.nan,
                  'Deployment Unknown': np.nan,
                  'Unknown Location': np.nan,
                  '': np.nan}, inplace=True)

    data['age'].replace({998: np.nan,
                         999: np.nan}, inplace=True)

    data['body_type'].replace(r'Unknown', np.nan, regex=True, inplace=True)

    data.dropna(inplace=True)

    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'Male' else 0)

    data['fatality'] = data['injury_severity'].apply(lambda x: 1 if 'Fatal' in x or 'Died' in x else 0)

    # Refactor land_use column to binary (0: Urban; 1: Rural)
    data['land_use'].replace({'Urban': 1,
                              'Rural': 0},
                             inplace=True)
    data.rename(columns={'land_use': 'land_use_urban'}, inplace=True)

    data['rollover'] = data['rollover'].apply(lambda x: 0 if x == 'No Rollover' else 1)

    def airbag_bin(x):
        if x == 'Not Deployed':
            return 0
        elif x == 'Not Applicable':
            return -1
        else:
            return 1

    data['air_bag_deployed'] = data['air_bag_deployed'].apply(airbag_bin)

    data['age'] = data['age'].astype(int)

    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    main()

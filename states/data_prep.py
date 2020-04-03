#!/usr/bin/env python3

import pandas as pd


def main():
    pd.set_option('display.max_columns', None)

    path = '../data/states'

    population_data = pd.read_excel(f'{path}/state_populations.xlsx', header=0, index_col='state_name')
    accident_fatality_data_2015 = pd.read_csv(f'{path}/accident_fatalities_2015.csv', header=0, index_col='state_name')
    accident_fatality_data_2016 = pd.read_csv(f'{path}/accident_fatalities_2016.csv', header=0, index_col='state_name')

    us_state_codes = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                      'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
                      'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
                      'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                      'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
                      'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
                      'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
                      'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
                      'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
                      'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI',
                      'Wyoming': 'WY', 'District of Columbia': 'DC', 'Puerto Rico': 'PR'}

    population_data['state_code'] = population_data.index.map(us_state_codes)
    accident_fatality_data_2015['state_code'] = accident_fatality_data_2015.index.map(us_state_codes)
    accident_fatality_data_2016['state_code'] = accident_fatality_data_2016.index.map(us_state_codes)

    accident_fatality_data_2015 = accident_fatality_data_2015.join(population_data[2015])
    accident_fatality_data_2016 = accident_fatality_data_2016.join(population_data[2016])

    accident_fatality_data_2015['accidents_per_100k'] = (accident_fatality_data_2015['accidents'] / accident_fatality_data_2015[2015]) * 10 ** 5
    accident_fatality_data_2016['accidents_per_100k'] = (accident_fatality_data_2016['accidents'] / accident_fatality_data_2016[2016]) * 10 ** 5

    accident_fatality_data = pd.concat([accident_fatality_data_2015, accident_fatality_data_2016])
    accident_fatality_data = accident_fatality_data.groupby(['state_name', 'state_number']).mean()
    accident_fatality_data.drop(['year', 2015, 2016], axis=1, inplace=True)

    accident_data = accident_data_prep()
    person_data = person_data_prep()

    data = accident_fatality_data.join(accident_data).join(person_data)
    data.reset_index(inplace=True)
    data['state_code'] = data['state_name'].map(us_state_codes)
    data.rename(columns={'manner_of_collision_Not Collision with Motor Vehicle in Transport (Not Necessarily in Transport for\n2005-2009)': 'manner_of_collision_Not Collision with Motor Vehicle in Transport'},
                inplace=True)
    data.info()

    data.to_csv(f'{path}/state_mean_accident_data.csv', index=False)


def accident_data_prep():
    path = '../data/accidents'
    data = pd.read_csv(f'{path}/accident_data_clean.csv', header=0)
    data.drop(['consecutive_number', 'vehicle_number', 'year', 'latitude', 'longitude', 'month'], axis=1, inplace=True)

    cat_cols = ['day_of_week', 'roadway_type', 'intersection', 'light_condition', 'atmospheric_conditions',
                'manner_of_collision', 'body_type', 'vehicle_conditions', 'part_of_day']
    binary_cols = ['land_use_urban', 'national_highway_system', 'previous_dwi_convictions', 'rollover',
                   'speeding_related', 'previous_speeding_convictions', 'driver_vision_obscured', 'is_weekend',
                   'multiple_vehicles', 'nonmotorist_involved', 'multiple_motorists', 'drunk_driver_involved',
                   'multiple_fatalities']
    numeric_cols = ['hour_of_day', 'vehicle_year', 'speed_limit']

    data = pd.get_dummies(data, columns=cat_cols)

    data = data.groupby(['state_name', 'state_number']).mean()

    # Remove columns where column mean value is below 0.01
    data = data.loc[:, data.mean() > 0.01]

    return data


def person_data_prep():
    path = '../data/persons'
    data = pd.read_csv(f'{path}/person_data_clean.csv', header=0)

    data.drop(['consecutive_number', 'vehicle_number', 'person_number', 'land_use_urban',
               'trafficway_type', 'manner_of_collision', 'body_type', 'rollover'], axis=1, inplace=True)

    cat_cols = ['person_type', 'injury_severity', 'seating_position', 'ejection', 'safety_equipment_use']
    binary_cols = ['sex', 'land_use_urban', 'air_bag_deployed', 'fatality']
    numeric_cols = ['age']

    data = pd.get_dummies(data, columns=cat_cols)

    data = data.groupby('state_number').mean()

    # Remove columns where column mean value is below 0.01
    data = data.loc[:, data.mean() > 0.01]

    return data


if __name__ == '__main__':
    main()

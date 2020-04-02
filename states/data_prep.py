#!/usr/bin/env python3

import pandas as pd


def main():
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
    accident_fatality_data = accident_fatality_data.groupby('state_name').mean().drop(['year', 2015, 2016], axis=1)
    accident_fatality_data['state_number'] = accident_fatality_data['state_number'].astype(int)

    accident_fatality_data.info()
    print()
    print(accident_fatality_data.describe())
    print()
    print(accident_fatality_data)


def accident_data():
    pass


def person_data():
    pass


if __name__ == '__main__':
    main()

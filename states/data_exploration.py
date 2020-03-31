#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

sns.set()

year = 2016
path = '../data/states'

accident_fatality_data = pd.read_csv(f'{path}/accident_fatalities_{year}.csv', header=0, index_col='state_name')
population_data = pd.read_excel(f'{path}/state_populations.xlsx', header=0, index_col='state_name')

us_state_codes = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
                  'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
                  'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
                  'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                  'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
                  'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
                  'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
                  'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
                  'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
                  'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
                  'District of Columbia': 'DC', 'Puerto Rico': 'PR'}

accident_fatality_data['state_code'] = accident_fatality_data.index.map(us_state_codes)
population_data['state_code'] = population_data.index.map(us_state_codes)

accident_fatality_data = accident_fatality_data.join(population_data[year])
accident_fatality_data['accidents_per_100k'] = (accident_fatality_data['accidents'] / accident_fatality_data[year]) * 10**5


def choropleth_map(location_series, data_series, title, legend_title):
    fig = go.Figure(
        data=go.Choropleth(
            locations=location_series,
            z=data_series,
            locationmode='USA-states',
            colorscale='Magma_r',
            autocolorscale=False,
            marker_line_color='white',
            colorbar_title=legend_title
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            y=0.9,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        geo=dict(
            scope='usa',
            projection=go.layout.geo.Projection(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )
    )

    fig.show()
    # fig.write_image('visualizations/choropleth_accidents_by_state.png')


choropleth_title = '2016 Fatal Accidents by State'
colorbar_title = 'Fatal Accidents'
choropleth_map(accident_fatality_data['state_code'],
               accident_fatality_data['accidents'],
               choropleth_title, colorbar_title)


choropleth_title = '2016 Fatal Accidents by State per 100k Population'
colorbar_title = 'Fatal Accidents per 100k'
choropleth_map(accident_fatality_data['state_code'],
               accident_fatality_data['accidents_per_100k'],
               choropleth_title, colorbar_title)

#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    path = '../data/states'
    data = pd.read_csv(f'{path}/state_mean_accident_data.csv', header=0)

    choropleth_title = 'Mean Fatal Accidents by State, 2015-2016'
    colorbar_title = 'Fatal Accidents'
    choropleth_map(data['state_code'],
                   data['accidents'],
                   choropleth_title, colorbar_title)

    choropleth_title = 'Mean Fatal Accidents by State per 100k Population, 2015-2016'
    colorbar_title = 'Fatal Accidents per 100k'
    choropleth_map(data['state_code'],
                   data['accidents_per_100k'],
                   choropleth_title, colorbar_title)

    data.set_index(['state_name', 'state_code', 'state_number'], inplace=True)
    rank_data = data.rank(ascending=False)
    rank_data.to_csv(f'{path}/state_rank_accident_data.csv', index=True)

    print(rank_data[rank_data['accidents_per_100k'] <= 5].sort_values('accidents_per_100k'))

    scatter_plot(data, 'land_use_urban')
    scatter_plot(data, 'safety_equipment_use_None Used')
    scatter_plot(data, 'light_condition_Dark – Not Lighted')

    plt.show()


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


def scatter_plot(df, x, y='accidents_per_100k'):
    ax = df.plot.scatter(x=x, y=y, figsize=(16, 8), s=120, linewidth=0)

    for k, v in df.iterrows():
        ax.annotate(k[1], (v[x], v[y]),
                    xytext=(10, -5), textcoords='offset points',
                    family='sans-serif', fontsize=14, color='darkslategrey')
    ax.get_figure().savefig(f'visualizations/{y}_vs_{x}.png')
    plt.show()


if __name__ == '__main__':
    main()

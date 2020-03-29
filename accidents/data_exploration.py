#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def main():
    path = '../data/accidents'
    data = pd.read_csv(f'{path}/accident_data_clean.csv', header=0)

    data.info()

    hours_conversion = {0: '12AM - 1AM', 1: '1AM - 2AM', 2: '2AM - 3AM', 3: '3AM - 4AM',
                        4: '4AM - 5AM', 5: '5AM - 6AM', 6: '6AM - 7AM', 7: '7AM - 8AM',
                        8: '8AM - 9AM', 9: '9AM - 10AM', 10: '10AM - 11AM', 11: '11AM - 12PM',
                        12: '12PM - 1PM', 13: '1PM - 2PM', 14: '2PM - 3PM', 15: '3PM - 4PM',
                        16: '4PM - 5PM', 17: '5PM - 6PM', 18: '6PM - 7PM', 19: '7PM - 8PM',
                        20: '8PM - 9PM', 21: '9PM - 10PM', 22: '10PM - 11PM', 23: '11PM - 12AM'}
    day_conversion = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                      'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    month_conversion = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12}

    data['hour_text'] = data['hour_of_day'].map(hours_conversion)
    data['day_number'] = data['day_of_week'].map(day_conversion)
    data['month_number'] = data['month'].map(month_conversion)

    hour_title = 'Percentage of Fatal Accidents by Hour of Day'
    hour_xlab = 'Hour of Day'
    hour_ylab = '% of Fatal Accidents'
    hour_fname = 'accidents_by_hour.png'

    bar_line_dual_plot(data, 'hour_of_day', 'hour_text',
                       hour_title, hour_xlab, hour_ylab, hour_fname, align='edge')

    day_title = 'Percentage of Fatal Accidents by Day of Week'
    day_xlab = 'Day of Week'
    day_ylab = '% of Fatal Accidents'
    day_fname = 'accidents_by_day.png'

    bar_line_dual_plot(data, 'day_number', 'day_of_week',
                       day_title, day_xlab, day_ylab, day_fname)

    month_title = 'Percentage of Fatal Accidents by Month'
    month_xlab = 'Month'
    month_ylab = '% of Fatal Accidents'
    month_fname = 'accidents_by_month.png'

    bar_line_dual_plot(data, 'month_number', 'month',
                       month_title, month_xlab, month_ylab, month_fname)

    line_plot(data)


def bar_line_dual_plot(data, num_col, str_col, title, xlab, ylab, fname, align='center'):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    data_num = data.groupby(num_col).size()
    data_num = (data_num / data_num.sum()) * 100.

    data_num.plot(ax=ax1)
    ax1.set_xlabel(xlab, fontsize=16)
    ax1.set_ylabel(ylab, fontsize=16)

    data_text = data.groupby(str_col).size().sort_values(ascending=False)
    data_text = (data_text / data_text.sum()) * 100.

    data_text.plot.bar(ax=ax2, align=align)
    ax2.tick_params('x', labelrotation=60)
    ax2.set_xlabel(xlab, fontsize=16)
    ax2.set_ylabel(ylab, fontsize=16)

    fig.subplots_adjust(top=0.25)
    fig.suptitle(title, fontsize=20)
    plt.show()
    fig.savefig(f'visualizations/{fname}')


def line_plot(data):
    data = data.groupby(['day_number', 'hour_of_day']).size()
    data = (data / data.sum()) * 100.
    ax = data.plot(figsize=(16, 8))

    ax.set_title('Percentage of Fatal Accidents by Day of Week & Hour of Day', fontsize=20)
    ax.set_xlabel('Day of Week & Hour of Day', fontsize=16)
    ax.set_ylabel('% of Fatal Accidents', fontsize=16)
    plt.show()
    ax.get_figure().savefig('visualizations/accidents_by_weekday_hour.png')


if __name__ == '__main__':
    main()

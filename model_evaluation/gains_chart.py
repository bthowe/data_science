import os
import sys
import time
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from bokeh.models import Slider
from bokeh.plotting import figure
from bokeh.io import output_file, show, save, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import HoverTool, ColumnDataSource, Select, Range1d, CategoricalColorMapper

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def model_lift_chart(y_test, y_prob, interval_width):
    df = pd.concat(
        [
            pd.Series(y_test).reset_index(drop=True),
            pd.Series(y_prob).reset_index(drop=True),
            pd.Series(
                pd.cut(y_prob, np.linspace(0, 1, (1. / interval_width + 1)), labels=range(int(1. / interval_width)),
                       right=True, include_lowest=True))
        ], axis=1)
    df.columns = ['y_test', 'y_prob', 'bin']
    df_gc = pd.DataFrame(np.linspace(0, 1 - interval_width, 1 / interval_width), columns=['Lower Bound'])
    df_gc['Upper Bound'] = np.linspace(0 + interval_width, 1, 1 / interval_width)
    df_gc['Population (bin)'] = df.groupby('bin')['y_test'].count()
    df_gc['% of Total Population (bin)'] = df_gc['Population (bin)'].apply(lambda x: x / float(len(df)))
    df_gc['Contact Population (bin)'] = df.groupby('bin')['y_test'].sum()
    df_gc['% of Total Contact Population (bin)'] = df_gc['Contact Population (bin)'].apply(
        lambda x: x / float(df['y_test'].sum()))
    df_gc['Population (cumulative)'] = df_gc['Population (bin)'].cumsum()
    df_gc['% of Total Population (cumulative)'] = df_gc['% of Total Population (bin)'].cumsum()
    df_gc['Contact Population (cumulative)'] = df_gc['Contact Population (bin)'].cumsum()
    df_gc['% of Total Contact Population (cumulative)'] = df_gc['% of Total Contact Population (bin)'].cumsum()
    df_gc['Lift'] = df_gc['% of Total Population (cumulative)'] / df_gc['% of Total Contact Population (cumulative)']
    df_gc = df_gc.fillna(0).replace(np.inf, np.nan).fillna(1).round(2)
    print df_gc

def _perfect_model_curve(y_test, y_prob):
    frac_of_successes = y_test.mean()

    def perfect_model(x):
        if x >= frac_of_successes:
            return 1
        else:
            return (1 / frac_of_successes) * x




def model_lift_plot(y_test, y_prob, name='Churned'):
    df = pd.concat(
        [
            y_test.reset_index(drop=True),
            pd.DataFrame(y_prob).reset_index(drop=True)
        ], axis=1
    )
    df.columns = ['target', 'y_prob']
    df.sort_values(['y_prob'], inplace=True)

    df['frac_of_total'] = 1
    df['frac_of_total'] = (df['frac_of_total'].cumsum() / len(df))
    df['frac_of_contacts'] = (df['target'].cumsum() / df['target'].sum())

    contact_frac = df['target'].sum() / float(len(df))

    def perfect_model(x):
        if x >= contact_frac:
            return 1
        else:
            return (1 / contact_frac) * x

    df['perfect_model'] = df['frac_of_total'].apply(perfect_model)
    df.reset_index(drop=True, inplace=True)


    outcome = savgol_filter(
        x=df['frac_of_contacts'],
        window_length=5,
        polyorder=3
    )
    p = np.polyfit(df['frac_of_total'], outcome, 15)  # use the smoother first
    # p = np.polyfit(df['frac_of_total'], df['frac_of_contacts'], 15)
    poly = np.poly1d(p)
    d_poly = poly.deriv()
    d2_poly = d_poly.deriv()

    model_preds = pd.concat([df['frac_of_total'],
                             df['frac_of_contacts'],
                             df['perfect_model'],
                             pd.Series(poly(df['frac_of_total'])),
                             pd.Series(d_poly(df['frac_of_total'])),
                             pd.Series(d2_poly(df['frac_of_total']))], axis=1)
    model_preds.columns = ['frac_of_total', 'frac_of_contacts', 'perfect_model', 'polynomial', 'derivative',
                           'derivative2']

    source = ColumnDataSource(data={
        'x': model_preds['frac_of_total'],
        'y': model_preds['frac_of_contacts'],
        'y_perf': model_preds['perfect_model'],
        'y_poly': model_preds['polynomial'],
        'y_der': model_preds['derivative'],
        'y_2der': model_preds['derivative2']
    })

    hover = HoverTool(names=['true_model'],
                      tooltips=[('Perfect Model', '@y_perf'), ('Estimate', '@y'), ('Random', '@x'),
                                ('Second Derivative', '@y_2der')], mode='vline')

    p = figure(title='Lift Curve', x_axis_label='Fraction of Leads', y_axis_label='Fraction {0}'.format(name),
               plot_height=400, plot_width=1000, tools=[hover, ])
    p.y_range = Range1d(0, 1)
    p.line(x='x', y='y', source=source, color='blue', legend='Model Prediction', name='true_model')
    p.line(x='x', y='x', source=source, color='red', legend='Random')
    p.line(x='x', y='y_perf', source=source, color='green', legend='Perfect Model')
    p.legend.location = 'bottom_right'

    dp = figure(title='Derivative of Lift Curve', x_axis_label='Fraction of Leads', y_axis_label='Derivative',
                plot_height=400, plot_width=1000,
                tools=[HoverTool(tooltips=[('Second Derivative', '@y_2der')], mode='vline'), ])
    dp.x_range = p.x_range
    dp.line(x='x', y='y_der', source=source, color='cyan', legend='Derivative of Lift Curve')
    dp.legend.location = 'bottom_right'

    layout = column(p, dp)
    show(layout)
    save(layout, filename='/Users/travis.howe/Downloads/lift_plot.html', title='Pretention Lift Plot')


if __name__ == '__main__':
    y_test = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)))[0]
    y_prob = pd.DataFrame(np.random.uniform(0, 1, size=(10, 1)))[0]
    model_lift_plot(y_test, y_prob)

#     todo: generalize
# bokeh todo: (1) model dropdown menu, (2) to get synchonized hovers for linked plots see: https://stackoverflow.com/questions/35983029/bokeh-synchronizing-hover-tooltips-in-linked-plotss
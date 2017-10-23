import os
import sys
import time
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import SVGAnalytics as svg
from random import randint
from yhat import Yhat, YhatModel

from bokeh.models import Slider
from bokeh.plotting import figure
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import HoverTool, ColumnDataSource, Select, Range1d, CategoricalColorMapper

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def model_lift_chart(X_test, y_test, best_model, interval_width):
    y_prob = best_model.predict_proba(X_test)[:, 1]

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

def model_lift_plot():
    df = joblib.load('cadence_data_pull.pkl')
    df['cadence_score'] = df['output'].apply(cadence_score)
    df['customer_id'] = df['output'].apply(customer_id)
    df['consider_again_at'] = pd.to_datetime(df['output'].apply(consider_again_at))
    df['call_attempt'] = df['input'].apply(call_attempt)
    df = df.merge(joblib.load('call_data_pull.pkl'), how='left', on='customer_id').dropna()
    df['minutes_between_lead_and_call'] = (pd.to_datetime(df['start_timestamp']) - pd.to_datetime(
        df['input'].apply(lead_creation_date))).dt.seconds / 60.
    df = df.loc[df['consider_again_at'] < df['start_timestamp']].sort_values(
        ['customer_id', 'consider_again_at', 'start_timestamp']).pipe(pare_non_pairs).pipe(target_create).pipe(
        feature_keep).sort_values('cadence_score', ascending=False)

    min_max = joblib.load('{0}_data_files/contact_lift_chart_min_max_{1}.pkl'.format(business.lower(), business))
    cs_min = min_max[0]
    cs_max = min_max[1]

    df['target'] = df['cadence_score'].apply(lambda x: (x - cs_min) / (cs_max - cs_min))

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

    p = np.polyfit(df['frac_of_total'], df['frac_of_contacts'], 15)
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

    p = figure(title='Lift Curve', x_axis_label='Fraction of Leads', y_axis_label='Fraction Contacted',
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

if __name__ == '__main__':
    pass
#     todo: generalize
# bokeh todo: (1) model dropdown menu, (2) to get synchonized hovers for linked plots see: https://stackoverflow.com/questions/35983029/bokeh-synchronizing-hover-tooltips-in-linked-plots
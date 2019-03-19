import numpy as np
import pandas as pd
from functools import partial
from bokeh.models import Slider
from bokeh.plotting import figure
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import HoverTool, ColumnDataSource, Select, Range1d, CategoricalColorMapper

class ModelCompareViz(object):
    """Allows the models passed in to be plotted and compared for a subset of the data points in the test set"""
    def __init__(self, X, y, models, models_names, covars, subsample_size=200):
        self.subsample_size = subsample_size  # todo: might be able to delete this (and instance in line below)
        self.index = np.random.choice(range(len(y)), self.subsample_size, replace=False)
        self.X = X.iloc[self.index].reset_index(drop=True)
        self.y = y.iloc[self.index].reset_index(drop=True)

        self.models = models
        self.models_names = models_names
        self.num_models = len(models)

        self.covars = covars
        self.thresh = 0.5
        self.model_preds = self._model_preds()

        self.source = ColumnDataSource(data=self.model_preds)
        self.color_mapper = CategoricalColorMapper(factors=[1, 0], palette=['orange', 'blue'])

        self.slider_dict = {}

    def _model_preds(self):
        model_preds = np.c_[np.arange(self.subsample_size).reshape(self.subsample_size, 1), self.y.values]
        for mod in self.models:
            model_preds = np.c_[model_preds, mod.predict_proba(self.X)[:, 1], mod.predict(self.X), np.ones((self.subsample_size, 1)) * self.thresh]

        df = pd.concat([pd.DataFrame(model_preds), self.X[self.covars]], axis=1)

        name_list = []
        for name in self.models_names:
            name_list.append('{0}_prob'.format(name))
            name_list.append('{0}_class'.format(name))
            name_list.append('{0}_thresh'.format(name))

        df.columns = ['index', 'actual_class'] + name_list + self.covars
        return df

    def _glyph(self, name):
        p = figure(
                    x_axis_label='Index Number',
                    y_axis_label='Predicted Probability',
                    plot_height=400,
                    plot_width=700,
                    tools=[HoverTool(tooltips=[(var, '@{0}'.format(var)) for var in self.covars]), 'box_select,reset'])
        p.circle(x='index', y='{0}_prob'.format(name), source=self.source, legend='actual_class',
                           color=dict(field='actual_class', transform=self.color_mapper), size=5)
        p.line(x='index', y='{0}_thresh'.format(name), source=self.source, color='red')
        p.y_range = Range1d(0, 1)
        return p

    def _slider(self, name):
        s = Slider(start=0, end=1, step=0.01, value=0.5, title='{0} Classification Threshold'.format(name))
        s.on_change('value', partial(self._update_plot, name=name))
        # self.slider_dict['name'] = s
        return s

    def _update_plot(self, attr, old, new, name):
        self.model_preds['{0}_thresh'.format(name)] = new
        self.source.data = self.model_preds.to_dict('series')

    def plot(self):
        layout = row(
            widgetbox([self._slider(name) for name in self.models_names]),
            column([self._glyph(name) for name in self.models_names])
        )
        # curdoc().add_root(layout)
        show(layout)

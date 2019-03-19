# Hal Varian: https://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.28.2.3, page 22.
'''
A good predictive model can be better than a randomly chosen control group, which is usually thought to be the gold
standard. To see this, suppose that you run an ad campaign in 100 cities and retain 100 cities as a control. After the
experiment is over, you discover the weather was dramatically different across the cities in the study. Should you add
weather as a predictor of the counterfactual? Of course! If weather affects sales (which it does), then you will get a
more accurate prediction of the counterfactual and thus a better estimate of the causal effect of advertising.
'''

import joblib


def data_create(data, critical_date):
    model_data = data.query('date < {}'.format(critical_date))
    treatment_assess_data = data.query('date >= {}'.format(critical_date))
    return model_data, treatment_assess_data


def treatment_effect_estimate(data, model):
    X = data
    y = X.pop('outcome')

    y_pred = model(X)

    return (y_pred - y).sum()


if __name__ == '__main__':
    date_of_event = '2018-04-02'
    data1, data2 = data_create(date_of_event)

#     train model using data1, and save as model.pkl
    model = joblib.load('model.pkl')

    print(treatment_effect_estimate(data2, model))

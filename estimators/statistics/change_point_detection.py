import sys
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def poisson_model():
    with pm.Model() as disaster_model:

        switchpoint = pm.DiscreteUniform('switchpoint', lower=year.min(), upper=year.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential('early_rate', 1)
        late_rate = pm.Exponential('late_rate', 1)

        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= year, early_rate, late_rate)

        disasters = pm.Poisson('disasters', rate, observed=disaster_data)

        trace = pm.sample(10000)

        pm.traceplot(trace)

    plt.show()

def binom_model(df):
    # todo: make sure this works ok
    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform('switchpoint', lower=df['t'].min(), upper=df['t'].max())

        # Priors for pre- and post-switch probability of "yes"...is there a better prior?
        early_rate = pm.Beta('early_rate', 1, 1)
        late_rate = pm.Beta('late_rate', 1, 1)

        # Allocate appropriate probabilities to periods before and after current
        p = pm.math.switch(switchpoint >= df['t'].values, early_rate, late_rate)

        p = pm.Deterministic('p', p)

        successes = pm.Binomial('successes', n=df['n'].values, p=p, observed=df['category'].values)

        trace = pm.sample(10000)

        pm.traceplot(trace)

        plt.show()

def uniform_model(df):
    """
    The switchpoint is modeled using a Discrete Uniform distribution.
    The observed data is modeled using the Normal distribution (likelihood).
    The priors are each assumed to be exponentially distributed.
    """
    alpha = 1.0 / df['score'].mean()
    beta = 1.0 / df['score'].std()

    t = df['t_encoded'].values

    with pm.Model() as model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=df['t_encoded'].min(), upper=df['t_encoded'].max())
        mu_1 = pm.Exponential("mu_1", alpha)
        mu_2 = pm.Exponential("mu_2", alpha)
        sd_1 = pm.Exponential("sd_1", beta)
        sd_2 = pm.Exponential("sd_2", beta)
        mu = pm.math.switch(switchpoint >= t, mu_1, mu_2)
        sd = pm.math.switch(switchpoint >= t, sd_1, sd_2)
        X = pm.Normal('x', mu=mu, sd=sd, observed=df['score'].values)
        trace = pm.sample(20000)

    pm.traceplot(trace[1000:], varnames=['switchpoint', 'mu_1', 'mu_2', 'sd_1', 'sd_2'])
    plt.show()

def data_generate():
    mu1 = .55
    sd1 = .1
    mu2 = .50
    sd2 = .1
    df = pd.DataFrame(norm.rvs(mu1, sd1, size=(50, 1)), columns=['score'])
    df['t_encoded'] = np.random.choice(range(10), size=(50, 1))
    df2 = pd.DataFrame(norm.rvs(mu2, sd2, size=(50, 1)), columns=['score'])
    df2['t_encoded'] = np.random.choice(range(10, 20), size=(50, 1))
    return df.append(df2).sort_values('t_encoded')

def data_generate2():
    mu1 = .60
    sd1 = .05
    mu2 = .40
    sd2 = .05
    df1 = pd.DataFrame(norm.rvs(mu1, sd1, size=(50, 1)), columns=['y'])
    df2 = pd.DataFrame(norm.rvs(mu2, sd2, size=(50, 1)), columns=['y'])
    df = df1.append(df2)
    df['ds'] = pd.date_range('2018-01-01', periods=100, normalize=True)
    return df

def fb_changepoint(df):
    """
    I'm trying to get fbprophet to indicate changepoints
    """
    lst = [0.6483603911394896, 0.6457161377104432, 0.6430718842813967, 0.6404276308523502, 0.6377833774227222, 0.6351391239930941, 0.6324948705634662, 0.629850617132599, 0.627206363701732, 0.624562110270865, 0.6219178568399979, 0.6192736034009921, 0.6166293499619863, 0.6139850965229804, 0.6113408430229693, 0.6086965895229581, 0.606052336022947, 0.6034080824278231, 0.6007638288326992, 0.5981195752375753, 0.5954753217287018, 0.5928310682198282, 0.5901868147109546, 0.5875425610400774, 0.5848983073692002, 0.5822540536983231, 0.5796098000274459, 0.5769655463156467, 0.5743212926038476, 0.5716770388920484, 0.5689598039202758, 0.5662425689485031, 0.5635253339767307, 0.5606741531796055, 0.5578229723824805, 0.5549717915853553, 0.551965016621816, 0.5489582416582766, 0.5459514666947373, 0.542809265192489, 0.5396670636902406, 0.5365248621879923, 0.533382660685744, 0.5302400989614969, 0.5270975372372498, 0.5239549755130026, 0.5208124135971544, 0.5176698516813059, 0.5145272897654576, 0.5113847278319311, 0.5082421658984048, 0.5050996039648784, 0.5019570420729639, 0.4988144801810494, 0.49567191828913487, 0.49252935638825884, 0.4893867944873828, 0.48624423258650684, 0.4831016706856308, 0.47995910884875714, 0.4768165470118834, 0.4736739851750098, 0.4705314233592992, 0.4673888615435886, 0.464246299727878, 0.46110373801587395, 0.45796117630386995, 0.454818614591866, 0.45167605291811547, 0.44853349124436487, 0.4453909295706144, 0.4422483690893287, 0.43910580860804294, 0.4359632481267572, 0.43282068764547144, 0.4296781271718466, 0.4265355666982217, 0.4233930062245969, 0.420250445787155, 0.4171078853497131, 0.41396532491227117, 0.4108227645090377, 0.40768020410580424, 0.40453764370257084, 0.4013950832993374, 0.3982525228961039, 0.39510996249287045, 0.39196740208963704, 0.3888248416864036, 0.3856822812831701, 0.38253972087993676, 0.37939716047670324, 0.37625460007346984, 0.3731120396702364, 0.3699694792670029, 0.36682691886376945, 0.363684358460536, 0.3605417980573025, 0.35739923765406906, 0.35425667725083565, 0.3511141168476022, 0.3479715564443688, 0.3448289960411353, 0.34168643563790185, 0.3385438752346684, 0.335401314831435, 0.3322587544282015, 0.32911619402496806, 0.32597363362173465, 0.3228310732185012, 0.3196885128152677, 0.3165459524120343, 0.31340339200880085, 0.3102608316055674, 0.30711827120233387, 0.3039757107991005, 0.30083315039586705, 0.2976905899926336, 0.29454802958940024, 0.2914054691861668, 0.2882629087829332, 0.28512034837969974, 0.2819777879764664, 0.2788352275732329, 0.27569266716999946, 0.27255010676676605, 0.2694075463635326, 0.26626498596029907, 0.2631224255570656, 0.25997986515383226, 0.2568373047505988, 0.25369474434736533, 0.2505521839441319, 0.2474096235408985, 0.24426706313766494, 0.24112450273443156, 0.2379819423311981, 0.23483938192796464, 0.23169682152473117, 0.2285542611214978, 0.22541170071826427, 0.2222691403150308, 0.21912657991179743, 0.21598401950856397, 0.2128414591053305, 0.20969889870209704, 0.20655633829886366, 0.20341377789563014, 0.20027121749239668, 0.1971286570891633]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['ds'], df['y'])
    ax.plot(df['ds'], lst[:100])
    plt.show()  # this doesn't indicate changepoints

    sys.exit()


    from fbprophet import Prophet

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=50)
    forecast = m.predict(future)  # I print the trend as a list and then plot it above

    future.rename(columns={'ds': 't'}, inplace=True)


if __name__ == '__main__':
    # df = data_generate()
    # uniform_model(df)
    # binom_model(df)


    fb_changepoint(data_generate2())


# change point in time from uniform distribution
# two binomial distribution (before and after change)


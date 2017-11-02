import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, acovf, ccf

pd.set_option('max_columns', 700)
pd.set_option('max_info_columns', 100000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

'''





% these are optional in opt_block_length_full.m, but fixed at default values here
KN=max(5,sqrt(log10(n)));
%mmax = ceil(sqrt(n));
mmax = ceil(sqrt(n))+KN;           % adding KN extra lags to employ Politis' (2002) suggestion for finding largest signif m
warning_flags=0;
round=0;
%Bmax = sqrt(n);                  % maximum value of Bstar to consider.
Bmax = ceil(min(3*sqrt(n),n/3));  % dec07: new idea for rule-of-thumb to put upper bound on estimated optimal block length

c=2;
origdata=data;
Bstar_final=[];

for i=1:k
   data=origdata(:,i);

   % FIRST STEP: finding mhat-> the largest lag for which the autocorrelation is still significant.
   temp = mlag(data,mmax);
   temp = temp(mmax+1:end,:);	% dropping the first mmax rows, as they're filled with zeros
   temp = corrcoef([data(mmax+1:end),temp]);
   temp = temp(2:end,1);

   % We follow the empirical rule suggested in Politis, 2002, "Adaptive Bandwidth Choice".
   % as suggested in Remark 2.3, setting c=2, KN=5
   temp2 = [mlag(temp,KN)',temp(end-KN+1:end)];		% looking at vectors of autocorrels, from lag mhat to lag mhat+KN
   temp2 = temp2(:,KN+1:end);		% dropping the first KN-1, as the vectors have empty cells
   temp2 = (abs(temp2)<(c*sqrt(log10(n)/n)*ones(KN,mmax-KN+1)));	% checking which are less than the critical value
   temp2 = sum(temp2)';		% this counts the number of insignif autocorrels
   temp3 = [(1:1:length(temp2))',temp2];
   temp3 = temp3(find(temp2==KN),:);	% selecting all rows where ALL KN autocorrels are not signif
   if isempty(temp3)
      mhat = max(find(abs(temp) > (c*sqrt(log10(n)/n)) )); % this means that NO collection of KN autocorrels were all insignif, so pick largest significant lag
   else
      mhat = temp3(1,1);	% if more than one collection is possible, choose the smallest m
   end
   if 2*mhat>mmax;
      M = mmax;
      trunc1=1;
   else
      M = 2*mhat;
   end
   clear temp temp2 temp3;


   % SECOND STEP: computing the inputs to the function for Bstar
   kk = (-M:1:M)';

   if M>0;
      temp = mlag(data,M);
      temp = temp(M+1:end,:);	% dropping the first mmax rows, as they're filled with zeros
      temp = cov([data(M+1:end),temp]);
      acv = temp(:,1);			% autocovariances
      acv2 = [-(1:1:M)',acv(2:end)];
      if size(acv2,1)>1;
         acv2 = sortrows(acv2,1);
      end
      acv = [acv2(:,2);acv];			% autocovariances from -M to M
      clear acv2;
      Ghat = sum(lam(kk/M).*abs(kk).*acv);
      DCBhat = 4/3*sum(lam(kk/M).*acv)^2;

% OLD nov07
%      DSBhat = 2/pi*quadl('opt_block_length_calc',-pi,pi,[],[],kk,acv,lam(kk/M));
%      DSBhat = DSBhat + 4*sum(lam(kk/M).*acv)^2;	% first part of DSBhat (note cos(0)=1)

% NEW dec07
      DSBhat = 2*(sum(lam(kk/M).*acv)^2);	% first part of DSBhat (note cos(0)=1)

      % FINAL STEP: constructing the optimal block length estimator
      Bstar = ((2*(Ghat^2)/DSBhat)^(1/3))*(n^(1/3));
      if Bstar>Bmax
         Bstar = Bmax;
      end
      BstarCB = ((2*(Ghat^2)/DCBhat)^(1/3))*(n^(1/3));

      if BstarCB>Bmax
         BstarCB = Bmax;
      end
      Bstar = [Bstar;BstarCB];
   else
      Ghat = 0;
      % FINAL STEP: constructing the optimal block length estimator
      Bstar = [1;1];
   end
   Bstar_final=[Bstar_final Bstar];
end
Bstar=Bstar_final;

%%%%%%%%%%%%%%%%%%%%%%%%
function lam=lam(kk)
%Helper function, calculates the flattop kernel weights
lam = (abs(kk)>=0).*(abs(kk)<0.5)+2*(1-abs(kk)).*(abs(kk)>=0.5).*(abs(kk)<=1);
'''


def mlags(series, lags):
    # todo: make the column names indicative of the lag number and the original column name as well
    df = pd.concat([series.shift(i) for i in xrange(lags + 1)], axis=1)
    df.columns = ['L{}'.format(i) for i in xrange(lags + 1)]
    return df


def opt_block_length(data):
    """This is a function taken from Andrew Patton (http://public.econ.duke.edu/~ap172/) to select the optimal (in the
    sense of minimising the MSE of the estimator of the long-run variance) block length for the stationary bootstrap or
    circular bootstrap. Code follows Politis and White, 2001, 'Automatic Block-Length Selection for the Dependent
    Bootstrap.'

    INPUTS:	data, an nxk pandas dataframe

    OUTPUTS: Bstar, a 2xk vector of optimal bootstrap block lengths, [BstarSB;BstarCB]

"""
    n, k = data.shape
    print n, k
    #
    KN = int(np.maximum(5, np.sqrt(np.log10(n))))
    # mmax = int(np.ceil(np.sqrt(n)) + KN)
    # warning_flags = 0  # todo: ?
    # round = 0
    # Bmax = np.ceil(np.minimum(3 * np.sqrt(n), n / 3))
    #
    c = norm.ppf(0.975)
    origdata = data
    # Bstar_final = []

    for i in xrange(0, k):

        mmax = 2  # todo: delete

        data = origdata.iloc[:, i]
        rho_k = acf(data, nlags=mmax)[1:]  # see note at this point in the code in http://public.econ.duke.edu/~ap172/ regarding sample correlations versus the acf as used here
        rho_k_crit = c * np.sqrt(np.log10(n) / n)

        ni_function = lambda x: np.sum((np.abs(rho_k) < rho_k_crit)[x: x + KN])
        num_insignificant = [ni_function(i) for i in xrange(mmax - KN + 1)]

        if any(num_insignificant == KN):
            mhat = num_insignificant.remove(KN)
            # mhat = [i for i in num_insignificant if i == KN][0]  # use
        else:
            if any(abs(rho_k) > rho_k_crit):
                lag_sig = [i for i in abs(rho_k) > rho_k_crit if i]
                k_sig = len(lag_sig)
                if k_sig == 1:
                    mhat = lag_sig
                else:
                    mhat = max(lag_sig)
            else:
                mhat = 1

        if 2 * mhat > mmax:
            M = mmax
        else:
            M = 2 * mhat

        kk = range(-M, M + 1)

        R_k = ccf(data)  # todo










if __name__ == '__main__':
    df = pd.DataFrame([[0.3004800, 0.4806089, -1.0232758, 0.9733925, 0.2688148, -0.5161701, -0.7270535, -1.7454589,
                        -0.1558424, -0.6287940],
                       [0.3814848, 0.3526421, 0.8240130, 0.4161907, -0.6108061, -2.3804401, 1.4487400, 0.7112594,
                        -0.4536392, 0.8210167]]).transpose()
    opt_block_length(df)



#

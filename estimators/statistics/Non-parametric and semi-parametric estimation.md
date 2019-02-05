https://ocw.mit.edu/courses/economics/
I'm interested in going through the causal NU and MIT slides/notes.


https://ocw.mit.edu/courses/economics/14-385-nonlinear-econometric-analysis-fall-2007/lecture-notes/
https://ocw.mit.edu/courses/economics/14-385-nonlinear-econometric-analysis-fall-2007/lecture-notes/notes_nonsemi.pdf
Nonparametric
* Empirical Distribution Function
* Kernel Density Estimator
* Nonparametric Regression
    1. Kernel Regression
        1. Tend to have poor bias at the boundaries  
    2. Series Regression (?)
    3. Locally Linear Regression
Semi-parametric (?)
* Models
    1. Binary Choice with Unknown Disturbance Distributions
    2. Censored Regression with Unknown Disturbance Distribution
    3. Partially Linear Regression
    4. Index Regression 
* Estimators
    1. Binary Choice
    2. Censored Regression
    
    
Splines: http://www.stat.cmu.edu/~larry/=sml/nonpar.pdf

Splines 
    1. over the m intervals, a spline is an estimated function of the data
    2. knots at the interval boundaries "tie" the splines together (in general, choosing knots is a tricky business)
    3. Types
        1. Regression splines
            1. Tend to display erratic behavior (i.e., high variance) at the boundaries of the input domain.  
        2. Natural splines
            1. Force the piecewise polynomial function to have a lower degree to the left of the leftmost knot and to the right of the rightmost knot.
        2. Smoothing splines
            1. 
            
            
Can I create some type of wrapper in python for quickly doing splines in R that will allow me to do everything as though it were sklearn

I don't think I understand how splines are used in prediction well.
- https://stat.ethz.ch/pipermail/r-help/2011-September/290434.html

For whatever reason, the servers were not starting on reboot. This fixed it.
```bash
sudo systemctl enable supervisor
```



It sounds appealing, but I, like you, am very skeptical the predictions would be better. 
The theoretical target variable is time elapsed between now until the "optimal" time to place the call.

What outcome could we use in the training of a model that would predict that?
We would need (1) attributes of a lead at an arbitrary point in time and (2) an indication of the optimal time to call.

For the former, we can use purchase or call times. 

Regarding the latter, we cannot use next call time because that call may not have been answered and so we would be predicting our behavior rather than the lead's.
It would have to be the call time of a call that was answered.
If calls were always made to leads at fixed intervals, then we could use the time between an answered call and an arbitrary point in time, of course conditional on all other covariates such as lead age and number of dials
If call time is randomly determined

#####*Hard to disentangle our behavior from the lead's
If call time is randomly determined does this work?

The question comes down to whether this interval is indicative of optimal wait time. 


If call time is randomly determined, the target thusly defined would allow for prediction of the time from the event to contact.
One potential complication is if call time is not determined randomly then any systematic bias in our calling could influence the predicted call time.
Further, what do we do with leads that never pick up? This means the prediction is biased downward.
    * This means survival analysis would be necessary


Recursive neural network
Any other survival model
 
What we want is a distribution of pick up over possible times. 



How is this regression scenario different than that of classification?
- In classification we want to know how the lead responds to our intervention. Ex post we determine when to call.
- In regression we want to know how long until we intervene and the lead responds. There are two variables in play. In order to get an accurate prediction of when they will answer our intervention must be not introduce bias.   
    - What we are predicting is, conditional on the covariates, when will they answer. 
    - This is different then "when should we call" which is what we actually care about.
    - The two models will give similar answers if (1) when we called in the data is random, (2) ...what else?






Two problems I see with the construction of the target:
1. What would be the starting time. In the holdout data the start time is obviously right now, but in the training and test
data, the start time would have to be either purchase time or a call time. 
2. The target would be interpretable as something like the time interval between the start time and when we plan on 
intervening via a call they would likely answer. 
However, in the data,   



One complication is the fact that half of every day we don't make calls. 
What do we do if the optimal predicted time is outside of business hours?



If we initially narrowed the scope of the problem to only today, then maybe.

What would the target be in the training data? 
* The time between the purchase and contact?
    * Complications:
        * 
        * What do we do if we call at that time and no one answers?
    * Overall, this seems like it would do far worse.
* The time between the last call and the contact?
* We could have models predicting how much time after purchase we should call first, second, and third, and another model
 predicting if this first era of call time elapsed, when is it optimal to call again?

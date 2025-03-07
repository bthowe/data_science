Objective: given the known behavior of the kicker, what is the optimal strategy of the goalie?

assumptions:
1. only one kicker (i.e., the shootout consists of only one shot. (relax this later on)
2. assume the score space is broken into six sectors (top left, top middle, top right, bottom left, bottom middle, bottom right)
3. the off-goal rate and save rate are as follows:
top-left: off goal rate: 9/59, 2/50
top-center: off goal rate: 9/40, 0/31
top-right: off goal rate: 7/42, 2/35
bottom-left: off goal rate: 9/127, 23/118
bottom-center: off goal rate: 0/46, 14/46
bottom-right: off goal rate: 5/119, 29/114
4. kickers strike in the "natural" direction 25% more frequently because they can kick it faster (how should I account for this? it probably makes it harder to stop, everything else equal...but it is also probably less accurate as well)
  -so I think I will let them kick in whichever location is optimal (the 25% would function as a constraint)
  -should I assume kick speed is fixed (in theory, this is something the kicker could manipulate and optimally calibrate)? I think so.
  - essentially, I'm saying, regardless of whether the kicker is left or right-footed, the only thing that matters are the percentages. A problem with this is, then, that there is nothing inherently different about the left and right sides. Therefore, I should just pool the data for the left and right, conditional on up/down
If the goalie guesses in the correct direction with what probability will he stop the ball?
assume the game is simultaneously played

The goalie has six actions just like the kicker: assume the probability of a stop differs depending on the action, conditional on the ball being kicked to that region.
If the ball is kicked to a different region, the probably of a stop is zero (but the probability of a miss is positive).

The equilibrium is a nash.


solve for the theoretical n.e.
can I get something coded up in tensorflow that computes this equilibrium?


Theoretical:
x



Empirical:
used reinforced learning
observe an outcome
give a point if the ball was blocked
it seems this could lead to the best response function for the keeper
but this isn't the essence of the nash equilibrium. it's not really a maximization problem, but rather conditional on the other guys actions, i have no incentive to deviate and visa versa. So finding his values that make me indifferent, and visa versa.
But data assumes something like repeated play and actions taken therein. So I could come up with a best response. 
So the idea of learning the nash equilibrium is not possible unless I have data on many different policies. 
But I can learn, assuming a single policy, what the best response would be. 




What if I used a multiarmed bandit...would this work?
It seems like it would. During the exploration phase, you randomly choose an "arm" and observe the outcome. Then you update. Or something like this.
 
 
 
How is a multiarmed bandit like reinforcement learning?







# Bandit
Two important features the bandit should achieve are (1) the probability of a win for each strategy and (2) the probability of a strategy being chosen.

I need a methodology that converges to the correct probability distribution of choosing a strategy. The Bayesian would not do this---the trials and wins would have to be the same for lower_right and lower_left.
Similarly, the epsilon greedy would not work because during the exploration stage anything could be chosen according to a uniform distribution, whereas during the exploitation phase the alternative with the highest win to trial ratio is chosen.
The UCB1 likewise does not work because it penalizes alternatives that have been chosen more, everything else equal. Thus, while the number of lower_right and lower_left chosen might be equal, the probability of one would depend on whatever values were realized recently. 

Using the softmax, we get a convergence to the following probability of keeper strategy choice distribution: [0.08664848 0.05617447 0.09048345 0.34544559 0.06878271 0.3524653 ]. I don't understand why it converged (more or less) to this.
changing the value of tau from .1 to .035 gives roughly the correct answer. The fraction of wins are approximately correct as well. However, the probability of choosing another strategy is too high.
Is tuning tau something that is even possible prior to running the bandit in the wild?

Can I think of an alternative solution?

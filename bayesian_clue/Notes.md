In some ways it is similar to the dice problem in that the value of the card is unknown and there are periodic updates.

# Likelihood 
* the likelihood says, conditional on that hand, what is the probability of that outcome? Hmmm, this might not be useful.
* If a hand has all three cards, or just one, the likelihood of "yes" is no different. Thus, the likelihood function is binary.


# TODO
1. make displaying a players possible hand easy
2. add the calculate cardinality of possible hands to the end of the card_reveal and uncertain_card_reveal methods

1. intelligent guessing: which guess will lead to the grestest uncertainty reduction? Can I calculate this?
    1. which card shows up the most in the remaining possible hands
    2. on my turn, how do I best rule it out?
2. should I make the tuple-strings just into lists?
3. the should I calculate actual posterior probabilities? This would necessitate combining information across hands.
4. using flask, create a UI that makes this easier to implement during an actual game
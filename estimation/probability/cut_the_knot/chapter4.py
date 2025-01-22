import sys
import math
import primefac
import itertools
import numpy as np
from functools import partial
from itertools import product
import numpy.polynomial.polynomial as poly

from optimization.mab import MAB
from optimization.mab_generalized import MAB as MAB_gen


def _monte_carlo(f, N=10_000):
    count = 0
    for _ in range(N):
        count += f()
    return count / N


def problem_4_1():
    """
    I like this one because it's counterintuitive, due to the probabilities of victory being one way but the optimal
    schedule of play the other (in that the more difficult opponent is faced twice). This is, generalized, like
    maximizing success by taking multiple cracks at the hard thing...all subject to the rules of the game constraints.
    
    An example that came to mind, though the scenarios are not perfectly analogous, are 'going for it' on fourth down
    versus kicking a field goal.
    
    Analytically, the possible outcomes are: 111, 110, 101, 011, 100, 010, 001, 000. The two that yield acceptance into
    the club are 111, 110, and 011, which have associated probabilities
    (1) p_N_G * p_N_T * P_N_G = 9/10 * 1/10 * 9/10 = 81/1000,
    (2) p_N_G * p_N_T * P_N_G = 9/10 * 1/10 * 1/10 = 9/1000,
    (3) p_N_G * p_N_T * P_N_G = 1/10 * 1/10 * 9/10 = 9/1000
    yielding a combined probability of 99/1000 for the first schedule and
    (1) p_N_T * p_N_G * P_N_T = 1/10 * 9/10 * 1/10 = 9/1000,
    (2) p_N_T * p_N_G * P_N_T = 1/10 * 9/10 * 9/10 = 81/1000,
    (3) p_N_T * p_N_G * P_N_T = 9/10 * 9/10 * 1/10 = 81/1000
    yielding a combined probability of 171/1000 for the second.
    """
    p_N_G = 0.9
    p_N_T = 0.1
    
    N = 1_000_000
    club_count1 = 0
    club_count2 = 0
    for _ in range(N):
        first_game = np.random.binomial(1, p_N_G, 1)[0]
        second_game = np.random.binomial(1, p_N_T, 1)[0]
        third_game = np.random.binomial(1, p_N_G, 1)[0]
        if (first_game * second_game == 1) or (second_game * third_game == 1):
            club_count1 += 1

        first_game = np.random.binomial(1, p_N_T, 1)[0]
        second_game = np.random.binomial(1, p_N_G, 1)[0]
        third_game = np.random.binomial(1, p_N_T, 1)[0]
        if (first_game * second_game == 1) or (second_game * third_game == 1):
            club_count2 += 1
    print(club_count1 / N)
    print(club_count2 / N)


def problem_4_2():
    """
    N block boxes in series. The input stays the same with probability p and switches with probability 1-p. What is the
    probability the input into the system is the same as output?
    """
    N = 10_000
    
    N_boxes = 50
    p = .01

    count_same = 0
    for _ in range(N):
        switch = 1
        for _ in range(N_boxes):
            if np.random.uniform(0, 1, 1)[0] < 1 - p:
                switch *= -1
        if switch == 1:
            count_same += 1
    print(count_same / N)
    
    
class Chess:
    def __init__(self):
        self.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.rows = list(range(1, 9))
        self.black_king_location = (np.random.choice(self.columns), 1)
        self.white_king_location = (np.random.choice(self.columns), 3)
        self.white_queen_location = self._initial_white_queen_location(self.black_king_location)
        
        self.possible_black_king_moves = self._k_moves(self.black_king_location)
        self.possible_white_king_moves = self._k_moves(self.white_king_location)
        self.possible_white_queen_moves = self._q_moves(self.white_queen_location)

    def _initial_white_queen_location(self, bk_location):
        while True:
            random_index = np.random.choice(len(list(product(self.columns, [1, 2]))))
            queen_location = list(product(self.columns, [1, 2]))[random_index]
            if bk_location != queen_location:
                return queen_location
    
    def _diagonal_iterator_queen(self, step, current_location_indeces):
        move_lst = []
        new_column_index = current_location_indeces[0]
        new_row_index = current_location_indeces[1]
        valid_square = True
        while valid_square:
            new_column_index = new_column_index + step[0]
            new_row_index = new_row_index + step[1]
            if (new_column_index < 0) or (new_column_index > 7) or (new_row_index < 0) or (new_row_index > 7):
                break
            else:
                move_lst.append((self.columns[new_column_index], self.rows[new_row_index]))
        return move_lst

    def _q_moves(self, current_location):
        moves_lst = list(product(current_location[0], self.rows))  # moving vertically
        moves_lst += list(product(self.columns, [current_location[1]]))  # moving horizontally

        current_location_indeces = (self.columns.index(current_location[0]), self.rows.index(current_location[1]))
        for step in product([-1, 1], [-1, 1]):
            moves_lst += self._diagonal_iterator_queen(step, current_location_indeces)  # down right step
        
        return [_ for _ in list(set(moves_lst)) if _ != current_location]  # Need to remove the current location because the black king could move there.

    def _iterator_king(self, step, current_location_indeces):
        new_column_index = current_location_indeces[0] + step[0]
        new_row_index = current_location_indeces[1] + step[1]
        
        if (new_column_index >= 0) and (new_column_index <= 7) and (new_row_index >= 0) and (new_row_index <= 7):
            return self.columns[new_column_index], self.rows[new_row_index]

    def _k_moves(self, current_location):
        move_lst = []
        current_location_indeces = (self.columns.index(current_location[0]), self.rows.index(current_location[1]))

        for step in product([-1, 0, 1], [-1, 0, 1]):
            if step == (0, 0):
                continue
            move_lst += [_ for _ in [self._iterator_king(step, current_location_indeces)] if _]  # list comprehension to not include the Nones
        
        return list(set(move_lst))
    
    def _check_checker(self):
        if not {self.black_king_location} - set(self.possible_white_king_moves + self.possible_white_queen_moves):  # currently in check
            return True
        return False
    
    def _next_move_check_checker(self):
        if not set(self.possible_black_king_moves) - set(self.possible_white_king_moves + self.possible_white_queen_moves):  # no possible moves
            return True
        return False
    
    def checkmate_checker(self):
        if self._check_checker() and self._next_move_check_checker():
            return True
        return False
    
    
def _dumb_helper_function():
    """
    I need this wrapper because without it the class does not reinstantiate every iteration of the for loop within
    _monte_carlo.
    """
    return Chess().checkmate_checker()


def problem_4_3():
    """
    I asked ChatGPT and it gave the wrong answer, but the denominator was the same as the book's solution (i.e., 92/960).
    With 1M simulations I'm getting 0.0957 compared to the book's answer of 0.0958. I'm pleased I got the answer correct
    but am slightly vexed it takes so long in questions of this sort, where there is an environment with rules like
    chess or tossing a coin on a table partitioned into squares.
    
    Why is this interesting to me? I don't know, but real life is complex.
    """
    print(_monte_carlo(_dumb_helper_function, N=10_000_000))


def problem_4_4a():
    """
    0.5
    I know there is value in doing this analytically, but dang this was fast. I played around a bit with the
    hyperparameters.
    """
    N = 10_000_000
    
    n_flips = 111
    p = 0.8
    print(np.sum(np.where(np.random.binomial(n_flips, p, N) % 2 == 0, 1, 0)) / N)


def _stupid_wrapper():
    return _even_headed(.75)

def _even_headed(p):
    n_heads = 0
    flips = 0
    while True:
        flips += 1
        if np.random.uniform() < p:
            n_heads += 1
        if n_heads == 2:
            return flips


def problem_4_4b():
    """
    What is the expected number of flips before getting an even number of heads the first time? The first appearance
    of an even number of heads in n tosses.
    
    It is a function of the probability. For example, if p=0.5, then the expected number is 4.
    """
    print(_monte_carlo(_stupid_wrapper, N=1_000_000))


def ping_pong():
    games = np.where(np.random.binomial(1, 2/3, 6), 1, -1)
    if np.where(np.abs(np.cumsum(games)) >= 2)[0].size == 0:
        return 6
    return np.where(np.abs(np.cumsum(games)) >= 2)[0][0] + 1
    
    
def problem_4_5():
    """
    The book's answer: 266 / 81 = 3.2839506172839505.
    
    Over 10M simulations, I got: 3.2841442. My solution was fast, relatively to the usual, since I didn't loop but used
    numpy arrays.
    
    np.cumsum, np.abs, np.where, np.random.binomial (which i probably use too much)
    """
    print(_monte_carlo(ping_pong, N=10_000_000))


def phone_number_expectation():
    forgotten_digit = np.random.randint(0, 10)
    return np.where(np.random.permutation(10) == forgotten_digit)[0][0]

def phone_number_prob(x):
    forgotten_digit = np.random.randint(0, 10)
    if np.where(np.random.permutation(10) == forgotten_digit)[0][0] + 1 <= x:
        return 1
    return 0

def problem_4_6():
    """
    The expected number of tries until success is 4.49641 (over 1M simulations).
    
    The probability it'll take at most 2 guesses is 0.2000859, and at most 3 is 0.3002489.
    
    A big gotcha is getting the indeces sorted out in these 'how many' problems, since python is 0 indexed but humans
    start counting things at 1.
    
    You'll also notice I started using the partial method from the functools module to avoid having to wrap.
    """
    ## expected number of tries until success
    # print(_monte_carlo(phone_number_expectation, N=1_000_000))
    
    ## probability of guessing by the second attempt
    print(_monte_carlo(partial(phone_number_prob, 2), N=10_000_000))
    print(_monte_carlo(partial(phone_number_prob, 3), N=10_000_000))
    
    
def _draw():
    return np.where(np.random.uniform(0, 1) < 0.5, 1, 0)

def _theoretical_urn_composition():
    return np.concatenate(([range(0, 122)], [range(121, -1, -1)]), axis=0).T


def likelihood(d):
    urn = _theoretical_urn_composition()
    
    p_same_color = (urn[:, 0] / 121) * (np.max((urn[:, 0] - 1, np.zeros(urn.shape[0])), axis=0) / 120) + \
        (urn[:, 1] / 121) * (np.max((urn[:, 1] - 1, np.zeros(urn.shape[0])), axis=0) / 120)
    p_diff_color = 1 - p_same_color
    
    if d:
        return p_same_color
    return p_diff_color


def problem_4_7():
    """
    This problem is kind of interesting because I'm not calculating a probability using a straightforward Monte Carlo.
    One approach might be to use a Monte Carlo to estimate the probability of drawing two balls of the same color for
    each of the various ball color counts (say, 10 white and 111 black, or 60 white and 61 black). This sounds extremely
    inefficient, and then testing a statistical difference between the probability and 0.5 (the probability from the
    problem description). Is there a better way?
    
    Another way, and one that is less computationally heavy, is using a Bayesian framework, estimating the probability
    distribution over counts of white and black balls: Since I know the probability of drawing two balls of the same
    color, each iteration I make a draw of whether the two balls are the same color or not. Then, conditional on the
    various possible counts of white and black balls I calculate the probability of a same color or not (depending on
    the draw, conditional on the count). I then multiply this by the prior, call this product the posterior, and update
    the prior.
    
    The above algorithm yielded 55 and 66 as the overwhelming favorite given a change tolerance of 0.000001. This is the
    book's answer as well.
    """
    prior = 1/122 * np.ones(122)  # this is the initial prior
    
    epsilon = 1  # precision tolerance...stopping criterion
    while epsilon > 0.000001:
        d = _draw()  # if equal to 1, then two of the same color were drawn; if 0, then otherwise
        posterior = likelihood(d) * prior
        posterior = posterior / np.sum(posterior)
        
        epsilon = np.max(np.abs(prior - posterior))
        prior = posterior
    
    print(posterior)
    print(np.max(posterior))
    print(_theoretical_urn_composition()[np.argmax(posterior), :])
    
    
def problem_4_7_chatgpt():
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define prior: uniform distribution over all possible values of x
    def prior(x):
        return 1 / 122  # Uniform prior for x in [0, 121]
    
    # Likelihood function: probability of drawing two balls of the same color given x
    def likelihood(x):
        total_balls = 121
        white_balls = total_balls - x
        same_color_prob = (x * (x - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))
        return same_color_prob
    
    # Bayesian update
    def posterior(x_vals, observed_prob=0.5):
        prior_vals = np.array([prior(x) for x in x_vals])
        likelihood_vals = np.array(
            [np.abs(likelihood(x) - observed_prob) for x in x_vals])  # Likelihood based on observation
        
        # Compute unnormalized posterior
        unnormalized_posterior = prior_vals * likelihood_vals
        normalized_posterior = unnormalized_posterior / np.sum(
            unnormalized_posterior)  # Normalize to get valid probabilities
        return normalized_posterior
    
    # Possible values of x (number of black balls)
    x_vals = np.arange(0, 122)
    
    # Calculate posterior
    posterior_vals = posterior(x_vals)
    
    # Find the MAP estimate (the x value with the highest posterior probability)
    map_estimate = x_vals[np.argmax(posterior_vals)]
    print(f"MAP estimate for the number of black balls: {map_estimate}")
    print(f"Number of white balls: {121 - map_estimate}")
    
    # Plot the posterior distribution
    plt.plot(x_vals, posterior_vals, label='Posterior Distribution')
    plt.axvline(map_estimate, color='red', linestyle='--', label=f'MAP: {map_estimate}')
    plt.xlabel('Number of Black Balls')
    plt.ylabel('Posterior Probability')
    plt.legend()
    plt.show()


def problem_4_7_chatgpt2():
    """
    (from chatgpt) Key Changes:
        Likelihood Function: We use a Gaussian likelihood (via norm.pdf) to model the likelihood of observing the given
        probability (0.5) based on the calculated probability for each 𝑥. The standard deviation 𝜎 is set to a small
        value (e.g., 0.01) to reflect the precision of the observation.
        
        Posterior Update: The posterior is calculated by multiplying the prior with this smoother likelihood function.
        
    Can we do this? I'm referring to line 387-ish: norm.pdf(same_color_prob, loc=observed_prob, scale=sigma)? It feels
    like we're cheating 'modeling the likelihood function as a Gaussian'. In my solution, I incorporate the known
    probability by drawing iteratively from the distribution, whereas chatgpt take (1) a difference and (2) a gaussian draw.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # Define prior: uniform distribution over all possible values of x
    def prior(x):
        return 1 / 122  # Uniform prior for x in [0, 121]
    
    # Likelihood function: Gaussian likelihood based on how close the calculated prob is to the observed 0.5
    def likelihood(x, observed_prob=0.5, sigma=0.01):
        total_balls = 121
        white_balls = total_balls - x
        same_color_prob = (x * (x - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))
        
        # Gaussian likelihood centered around the observed probability with a small standard deviation (sigma)
        return norm.pdf(same_color_prob, loc=observed_prob, scale=sigma)
    
    # Bayesian update
    def posterior(x_vals, observed_prob=0.5, sigma=0.01):
        prior_vals = np.array([prior(x) for x in x_vals])
        likelihood_vals = np.array([likelihood(x, observed_prob, sigma) for x in x_vals])
        
        # Compute unnormalized posterior
        unnormalized_posterior = prior_vals * likelihood_vals
        normalized_posterior = unnormalized_posterior / np.sum(
            unnormalized_posterior)  # Normalize to get valid probabilities
        return normalized_posterior
    
    # Possible values of x (number of black balls)
    x_vals = np.arange(0, 122)
    
    # Calculate posterior
    posterior_vals = posterior(x_vals)
    # print(posterior_vals)
    
    # Find the MAP estimate (the x value with the highest posterior probability)
    map_estimate = x_vals[np.argmax(posterior_vals)]
    print(f"MAP estimate for the number of black balls: {map_estimate}")
    print(f"Number of white balls: {121 - map_estimate}")


def two_cards():
    deck = np.random.permutation(['red'] * 26 + ['black'] * 26)
    
    my_cards = 0
    his_cards = 0
    for pair in zip(deck[::2], deck[1::2]):
        if pair == ('black', 'black'):
            my_cards += 2
        elif pair == ('red', 'red'):
            his_cards += 2
    # print(my_cards, his_cards)
    if my_cards <= his_cards:
        return -1
    return (my_cards - his_cards) * 3 - 1


def problem_4_8():
    """
    I would never have more cards than he, so will never recover my $1 spent to play.
    
    An interesting question is, even though my iterations continue yielding the same answer, how can I be certain
    another answer isn't possible? This is the turkey problem.
    """
    print(_monte_carlo(two_cards))


def problem_4_9():
    """
    Beautiful. I love the book's visual, which reminds me of Wolfram's A New Kind of Science.
    """
    lights = np.ones(100)
    
    for multiple in range(2, 101):
        index = np.arange(1, 101) % multiple == 0
        lights[index] += 1
        lights = lights % 2
    print(lights)
    print(np.mean(lights))


def multiple_of_three():
    bag = np.random.permutation([1, 2, 3, 4])
    number = bag[0] * 100 + bag[1] * 10 + bag[2]
    return 1 if number % 3 == 0 else 0
    

def problem_4_10():
    """
    Over 10M simulations I estimate the probability to be 0.4997536. The book's answer is 1/2.
    """
    print(_monte_carlo(multiple_of_three, N=10_000_000))
    
    
def problem_4_11():
    """
    todo: I wasn't able to do this one.
    This one is not easy to do. I'm having a hard time creating a symmetric matrix that uses all of the numbers in each
    column.
    """
    
    # [
    #     np.random.choice(range(1, 8), replace=False, size=(7, 1))
    # ]

    mat = np.zeros((7, 7))
    col1 = list(np.random.choice(range(1, 8), replace=False, size=7))
    mat[:, 0] = col1
    mat[0, 1:] = col1[1:]
    
    col2 = [num for ind, num in enumerate(range(1, 8)) if (num not in [col1[1]]) and (num != col1[ind])]
    mat[1:, 1] = col2
    mat[1, 2:] = col2[1:]

    print(mat)


    sys.exit()
    
    col1 = list(np.random.choice(range(1, 8), replace=False, size=7))
    col2 = [col1[1]] + [num for num in range(1, 8) if num != col1[1]]
    col3 = [col1[2]] + [col2[1]] + [num for num in range(1, 8) if num not in [col1[2], col2[1]]]
    col4 = [col1[3]] + [col2[2]] + [col3[1]] + [num for num in range(1, 8) if num not in [col1[3], col2[2], col3[1]]]
    print(col1, col2, col3, col4)
    
    print(np.concatenate([[col1], [col2], [col3], [col4]], axis=0).T)
    
    sys.exit()
    count = 0
    while True:
        count += 1
        square = np.concatenate([np.random.choice(range(1, 8), replace=False, size=(7, 1)) for _ in range(7)], axis=1)
        if np.array_equal(square, square.T):
            if np.array_equal(np.arange(1, 8), np.sort(np.diagonal(square))):
                return 1
            else:
                return 0
        print(square)
        print(np.array_equal(square, square.T), count)


def temperature():
    C = np.random.uniform(15, 25)
    F = lambda x: (9/5) * x + 32
    if round(F(round(C, 0)), 0) != round(F(C), 0):
        return 1
    return 0

def problem_4_12():
    """
    I enjoy a nice plot encapsulating the data generating process, which is the solution has.
    
    *** This is one reason for constructing at least a partial analytical solution. I'm just being a dumb data scientist
    with my simulations.
    
    *** This is also a nice rational for telling my children to draw pictures when solving problems in their math...
    it isn't cheating, but helps us understand how the data was generated.
    """
    print(_monte_carlo(temperature, N=1_000_000), 4/9)


def roots():
    coefs = [1] + list(np.random.uniform(-5, 5, size=2))
 
    # roots = np.roots(coefs)
    roots = poly.polyroots(coefs[::-1])
    
    if np.all(~np.iscomplex(roots)):
        if np.all(np.where(roots > 0, 1, 0)):
            if roots[0] != roots[1]:
                return 0
    return 1


def problem_4_13():
    """
    np.roots, np.iscomplex, np.all, np.where, np.polynomial.polynomial.polyroots
    
    Weirdly, np.where gives a lame one element array if I don't specify arguments: np.where(condition) does return an
    array with True and False.
    
    todo: Why is my answer off, by about 0.025: 0.89771 vs 0.92361?
    """
    print(_monte_carlo(roots, N=1_000_00))
    print((7 + np.sqrt(5)) / 10)
    


def problem_4_14():
    """
    
    """
    
    
    
def likelihood2(n, draw):
    class_compo = np.array(list(zip(range(n + 1), range(n, -1, -1))))
    
    if draw == 'same':
        # boy and boy plus girl and girl
        p_b = (class_compo[:, 0] * (class_compo[:, 0] - 1)) / (n * (n-1))
        p_g = (class_compo[:, 1] * (class_compo[:, 1] - 1)) / (n * (n-1))
        p = p_b + p_g
    else:
        # boy and girl plus girl and boy
        p = (class_compo[:, 0] * (class_compo[:, 1])) / (n * (n-1)) * 2
    return p


def _back_test(b, g):
    group = list(np.random.permutation(['b'] * b + ['g'] * g))
    if group[0] == group[1]:
        return 1
    return 0
    
    
def problem_4_15():
    """
    Back test to make sure the MAP is plausible. Why? because I assume n=40 to begin with but this might not yield
    'exactly an even chance that they are of the same sex.' This means calculating the MAP and then estimating the
    probability the two draws are the same via Monte Carlo.
    
    Somewhat interestingly, if N=40, then the MAP equals 17 boys and 23 girls. This, in turn, yields a probability of
    0.498876 that the two draws are of the same sex (over 10M simulations). The probability calculated analytically is
    
    (17/40)*(16/39) + (23/40)*(22/39) = 0.49871794871794867
    
    which is not equal to 0.5. But is 0.4987258 close enough to 0.5 that I would have concluded they are equal? Probably,
    which would be a mistake.
    
    ****More generally, how do I know if I have done enough simulations to conclude equality?
    
    Another solution, apart from the analytical one the book offers, is doing a search using the likelihood2 function
    across different values of N.
    """
    n = 40
    prior = 1  # assuming uninformative initial prior
    
    N = 10_000
    for _ in range(N):
        draw = list(np.random.permutation(['same', 'different'])).pop()
        posterior = likelihood2(n, draw) * prior
        prior = posterior / np.sum(posterior)
    
    b = np.argmax(prior)
    g = n - b
    print(b, g)
    print(_monte_carlo(partial(_back_test, b, g), N=10_000_000))
    # _back_test(np.argmax(prior), n - np.argmax(prior))


def problem_4_15_search():
    """
    Searching. Kind of cheating because I calculate probabilities within the likelihood  function, I guess.
    
    But also, my solution above, I'd have to search across values of n. Blah.
    """
    for n in range(2, 10001):
        a = np.where(likelihood2(n, 'same') == 0.5)[0]
        if a.size != 0:
            print(a)


class Gladiator:
    def __init__(self, a, b, strategy_a):
        self.a = a
        self.b = b
        self.strategy_a = strategy_a
    
    def fight(self, gladiator_a, gladiator_b):
        if np.random.uniform() < self.a[gladiator_a] / (self.a[gladiator_a] + self.b[gladiator_b]):
            self.a[gladiator_a] += self.b[gladiator_b]
            del self.b[gladiator_b]
        else:
            self.b[gladiator_b] += self.a[gladiator_a]
            del self.a[gladiator_a]
            
    def battle(self):
        while self.a and self.b:
            self.fight(self.strategy_a(self.a), min(self.b, key=self.b.get))
            # self.fight(self.strategy_a(self.a), np.random.choice(list(self.b.keys())))  # random strategy against random strategy
        if self.a:
            return 1
        return 0


def _strategy(dict):
    return max(dict, key=dict.get)
    # self.fight(max(self.a, key=self.a.get), np.random.choice(list(self.b.keys())))  # always choosing the strongest against a random strategy


def _gladiator():
    a_n = 2
    a = {f'a_{gladiator}': np.random.randint(1, 11) for gladiator in range(1, a_n + 1)}
    b_n = 2
    b = {f'b_{gladiator}': np.random.randint(1, 11) for gladiator in range(1, b_n + 1)}
    return Gladiator(a, b, _strategy).battle()


def problem_4_16():
    """
    How would I test a strategy and a best-response?
    
    Conduct a series of monte carlos in order to generate a distribution for a given strategy. Do this for at least one
    other strategy, and then compare distributions.
    
    I'm not seeing any difference when changing strategies, conditional on the team size and strength distributions
    
    ***I like the idea of this problem as a bandit, where a strategy is an arm potentially yielding differentially.
    """
    print(_monte_carlo(_gladiator, N=100_000))
    

class WinByTwo():
    def __init__(self, p, q):
        self.p = p
        self.q = q
        
        self.a_wins = 0
        self.b_wins = 0
        
    def game(self):
        outcome_prob = np.random.uniform()
        if outcome_prob < self.p:
            self.a_wins += 1
        elif self.p <= outcome_prob < self.p + self.q:
            self.b_wins += 1
        
    def series(self):
        while np.abs(self.a_wins - self.b_wins) < 2:
            self.game()
            # print(self.a_wins, self.b_wins)
        if self.a_wins > self.b_wins:
            return 1
        return 0


def _series_simulate():
    p = 0.2
    q = 0.7
    return WinByTwo(p, q).series()
    
    
def problem_4_17():
    """
    I don't love these types of problems because they seem pathological vis-a-vis reality.
    """
    print(_monte_carlo(_series_simulate))


def stones():
    stones_n = 14
    
    # player 1, turn 1
    stones_n -= np.random.choice(range(1, 6))
    
    # player 2, turn 1
    if stones_n < 12:
        stones_n -= (stones_n - 6)
    elif stones_n > 12:
        stones_n -= 1
    else:
        pass
    print(stones_n)


class Stones:
    def __init__(self, initial_action, second_action=None):
        self.initial_action = initial_action
        self.second_action = second_action
        
        self.stone_count = 14 - initial_action
        self.turn = 2
        
        self.player1_turn_count = 2
        
    def iteration(self):
        if (self.turn == 2) and (self.stone_count % 6 == 0):
            self.stone_count -= 1
            self.turn = 1
        elif (self.turn == 2) and (self.stone_count % 6 != 0):
            self.stone_count -= self.stone_count % 6
            if self.stone_count != 0:
                self.turn = 1
        elif (self.turn == 1) and (self.stone_count > 5):
            self.stone_count -= np.random.choice(range(1, 6))
            self.turn = 2
        elif (self.turn == 1) and (self.stone_count <= 5):
            self.stone_count = 0
        
    def iteration2(self):
        if (self.player1_turn_count == 2) and (self.turn == 1):
            self.stone_count -= self.second_action
            self.player1_turn_count += 1
            self.turn = 2
        else:
            self.iteration()


    def game(self):
        while self.stone_count > 0:
            self.iteration()
        if self.turn == 1:
            return 1
        return 0

    def game2(self):
        while self.stone_count > 0:
            self.iteration2()
        if self.turn == 1:
            return 1
        return 0
        
        
def _stones_wrapper():
    return Stones().game()


def problem_4_18():
    """
    On your turn you want to be left with five or fewer stones. So your opponent needs to be left with six. This means
    you need to be left with between seven and eleven stones. Which means your opponent must be left with twelve stones.
    If left with 13 then death. Which means you must initially take two. In summary, the optimal strategy is leaving
    your opponent with twelve or six stones. If twelve, then on your turn remove how ever many necessary to leave him
    with six.
    
    1/5 chance of getting it right on the first turn. Conditional on doing so, there is another 1/5 chance of getting
    it right on the second turn. So the answer is 1/25.
    
    
    Is there a way of learning the optimal strategy? I was thinking that I could iterate to by using a MAB, the
    strategies being removing 1, 2, 3, 4, or 5 stones.
    """
    ## this solves the problem
    # print(_monte_carlo(_stones_wrapper, N=1_000_000))
    
    ## an attempt to iterate to the optimal strategy
    n = 101
    param_space = np.linspace(0, 1, n)
    arms_initial_dist_dict = {
        1: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))},
        2: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))},
        3: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))},
        4: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))},
        5: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))},
    }
    from optimization.mab import MAB
    b = MAB(arms_initial_dist_dict)
    
    count = 0
    while count < 100_000:
        arm = b.choose_arm()
        obs = Stones(arm[1]).game()
        b.posterior_update(arm[1], obs, arm[0])
        count += 1
    # print(b.draws)
    print(b.posteriors)
    print(b.maps)
        # input()
#     todo: let's combine with another round of guesses. expand the strategy space to twenty-five
    
def problem_4_18_full():
    """
    How could I do this without assuming player two removes only 1 if backed against a wall? Paths...instead of arms I'd
    model paths.
    
    How is this different than a more traditional setup that involves data on paths (plays by both players) and then
    the estimation of a model like a logistic regression? They have to be similar if not the same.
    """
    strategy_space = list(itertools.product(range(1, 6), range(1, 6)))
    n = 101
    param_space = np.linspace(0, 1, n)
    arms_initial_dist_dict = {strategy: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))} for strategy in strategy_space}
    b = MAB(arms_initial_dist_dict)
    
    count = 0
    while count < 500_000:
        arm = b.choose_arm2()  # would be better to call this strategy rather than arm
        obs = Stones(arm[1], arm[2]).game2()
        b.posterior_update2(arm[1:], obs, arm[0])
        count += 1
    print(b.draws)
    print(b.posteriors)
    print(b.maps)


def _monte_carlo2(f, N=10_000):
    count = 0
    N_count = 0

    while N_count < N:
        funcy = f()
        if funcy is not None:
            count += funcy
            N_count += 1
    return count / N


def _dice():
    rolls = np.random.randint(1, 7, size=3)
    if np.sum(rolls[:2]) == rolls[2]:
        if 2 in rolls:
            return 1
        return 0

    
def problem_4_19():
    """
    Things get too big after the 63rd power. Do I just throw up my hands and say, 'sixty-three numbers are good enough'?
    This yields the same approximation as the second analytical solution in the book. A bit unsatisying. Numbers can
    get too big or too small. What to do, hmmm.
    """
    a = [str(num)[0] for num in 2 ** np.array(range(63))]
    print(np.mean(np.where(np.array(a) == '1', 1, 0)))
    
    
def problem_4_20():
    """
    not a fan of these number theory questions. Well, maybe I am.
    """


def problem_4_24():
    """
    Ah, let's assume the colors are randomly assigned to a head, in equal probability. One randomly chosen person simply
    guessing the color yields a probability of winning equal to 0.5.
    
    I thought about the strategy presented as solution 4 in the book. This can't be right. So counter intuitive it is
    unbelievable. Essentially, my hat color is red or blue with equal probability. But if I condition my strategy based
    on what I see in front me, it can be greater than 1/2. It is remarkable. How can I make this type of think intuitive?
    It is kind of like the Monte Hall problem. Changing, so to speak, can increase your chance of winning.
    
    If you see different colors, guess blue; pass otherwise (3/8 probability of success)
    If you see the same colors, guess the same; pass otherwise (2/8)
    If you see the same colors, guess the opposite; pass otherwise (6/8)
    
    rrr     ppp x   !!! !   xxx x
    rrb     xxp x   ppx x   pp! !
    rbr     xpx x       x   p!p !
    brr     pxx x       x   !pp !
    rbb     p!! !       x   !pp !
    brb     !p! !       x   p!p !
    bbr     !!p !       x   pp! !
    bbb     ppp x       !   xxx x
    
    How does one even come up with this? Trial and error, I suppose.
    """
    
    
def problem_4_25():
    """
    What's the best way to incorporate a constraint into a Monte Carlo? One option is running it and throwing out
    simulations that don't comply. The downside is inefficiencies. The upside is the fact that I don't have to embed as
    much into my code.
    
    Also, the monte carlo function from above will need to be modified.
    
    I'm getting 0.531 ~ 8/15.
    """
    print(_monte_carlo2(_dice, N=100_000))
    
    
class Truel:
    def __init__(self, p, q, r, first_game):
        self.p = p
        self.q = q
        self.r = r
        self.first_game = first_game
        
        self.A_wins = 0
        self.B_wins = 0
        self.C_wins = 0
    
    def _winner(self, winner):
        if winner == 'A':
            self.A_wins += 1
        elif winner == 'B':
            self.B_wins += 1
        else:
            self.C_wins += 1
    
    def _game(self, match_up):
        if match_up == ('A', 'B'):
            prob = self.p
        elif match_up == ('B', 'C'):
            prob = self.q
        elif match_up == ('A', 'C'):
            prob = self.r
        
        if np.random.uniform() <= prob:
            winner = match_up[0]
        else:
            winner = match_up[1]
        self._winner(winner)
        return winner

    def _next_match_up(self, winner, previous_match_up):
        return tuple([player for player in ['A', 'B', 'C'] if (player == winner) or player not in previous_match_up])
        
    def truel(self):
        match_up = self.first_game
        while (self.A_wins < 2) and (self.B_wins < 2) and (self.C_wins < 2):
            match_up = self._next_match_up(self._game(match_up), match_up)
        if self.B_wins == 2:
            return 1
        else:
            return 0
    
    
def _truel_wrapper():
    p = 0.6  # A beats B
    q = 0.6  # B beats C
    r = 0.9  # A beats C
    
    return Truel(p, q, r, ('A', 'B')).truel()
        
def problem_4_26():
    """
    How can I do this as a MAB? Actually, the MAB context is different. It's one of repeated plays and imperfect
    information.
    """
    ## iterating over the three possible first matches, B against C seems most advantageous for B
    print(_monte_carlo(_truel_wrapper, N=100_000))
    
    
    ## MAB
    strategy_space = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    
    n = 101  # because I'm estimating the bernoulli parameter p
    param_space = np.linspace(0, 1, n)
    strategy_initial_dist_dict = {strategy: {'param_space': param_space, 'initial_prior': dict(zip(param_space, [1 / n] * n))} for strategy in strategy_space}
    
    # b = MAB_gen(strategy_initial_dist_dict)
    #
    # count = 0
    # while count < 500_000:
    #     strategy = b.choose_arm()  # would be better to call this strategy rather than arm
    #     print(strategy); sys.exit()
    #     obs = Stones(arm[1], arm[2]).game2()
    #     b.posterior_update2(arm[1:], obs, arm[0])
    #     count += 1
    # print(b.draws)
    # print(b.posteriors)
    # print(b.maps)


def _crossings():
    crossings = np.random.randint(5, 12 * 60, size=1000)
    if np.any(np.where((6 * 60 < crossings) & (crossings <= 6 * 60 + 5), 1, 0)):
        return 0
    return 1


def problem_4_28():
    """
    What is 'a day of 12 hours'?
    
    I'm seeing, over 10M simulations something closer to 0.00090 than 0.00096.
    """
    print(_monte_carlo(_crossings, N=10_000_000))


def main():
    # problem_4_1()
    # problem_4_2()
    # problem_4_3()
    # problem_4_4a()
    # problem_4_4b()
    # problem_4_5()
    # problem_4_6()
    # problem_4_7()
    # problem_4_7_chatgpt()
    # problem_4_7_chatgpt2()
    # problem_4_8()
    # problem_4_9()
    # problem_4_10()
    # problem_4_11()
    # problem_4_12()
    # problem_4_13()
    # problem_4_15()
    # problem_4_15_search()
    # problem_4_16()
    # problem_4_17()
    # problem_4_18()
    # problem_4_18_full()
    # problem_4_19()
    # problem_4_25()
    # problem_4_26()
    problem_4_28()


if __name__ == '__main__':
    main()
# I'd like to understand more about how much you win when you bet conditional on odds, for 4.23
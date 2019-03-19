strats = ['upper_left', 'upper_middle', 'upper_right', 'lower_left', 'lower_middle', 'lower_right']

# keeper
p_block = {
    'upper_left': .7,
    'upper_middle': .95,
    'upper_right': .7,
    'lower_left': .75,
    'lower_middle': .95,
    'lower_right': .75
}  # conditional on the same keeper and kicker action

# kicker
p_shot = {
    'upper_left': .135,
    'upper_middle': .05,
    'upper_right': .135,
    'lower_left': .31,
    'lower_middle': .06,
    'lower_right': .31
}
p_miss = {
    'upper_left': 16 / 101,
    'upper_middle': 9 / 40,
    'upper_right': 16 / 101,
    'lower_left': 14 / 246,
    'lower_middle': 0 / 46,
    'lower_right': 14 / 246
}  # conditional on the kicker action


# this is the probability of a success given the action conditional on the above distributions
{
    'upper_left': .169,
    'upper_middle': .126,
    'upper_right': .169,
    'lower_left': .309,
    'lower_middle': .146,
    'lower_right': .309
}

# so the ratio of wins to trials should converge to these numbers

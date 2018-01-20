import sys
import itertools
import pandas as pd
from collections import defaultdict

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class PlayerHand(object):
    def __init__(self, deck, name, size_of_hand):
        self.deck = deck
        self.name = name
        self.size_of_hand = size_of_hand

        self.rooms = ['Study', 'Kitchen', 'Hall', 'Conservatory', 'Lounge', 'Ballroom', 'Dining Room', 'Library', 'Billiard Room']
        self.suspects = ['Plum', 'White', 'Scarlet', 'Green', 'Mustard', 'Peacock']
        self.weapons = ['rope', 'dagger', 'wrench', 'pistol', 'candlestick', 'lead pipe']

        self.possible_hands = self._possible_hands_create()

    def _possible_hands_create(self):
        all_combinations = [str(tuple(cards)) for cards in itertools.combinations(self.deck, self.size_of_hand)]

        df = pd.DataFrame(all_combinations, columns=['hand_{}'.format(self.name)])
        df['posterior_prob'] = 1 / len(df)
        for card in self.rooms + self.suspects + self.weapons:
            df[card] = 0
            df.loc[df['hand_{}'.format(self.name)].str.contains(card), card] = 1
        return df

    def posterior_update(self, yes_no, card):
        if yes_no == 'yes':
            self.possible_hands.loc[~self.possible_hands['hand_{}'.format(self.name)].str.contains(card), 'posterior_prob'] = 0
            self._normalize(self.possible_hands)
        else:
            for c in card:
                self.possible_hands.loc[self.possible_hands['hand_{}'.format(self.name)].str.contains(c), 'posterior_prob'] = 0
                self._normalize(self.possible_hands)

    def _normalize(self, df):
        df['posterior_prob'] = df['posterior_prob'] / df['posterior_prob'].sum()

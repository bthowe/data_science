import sys
import itertools
import functools
import pandas as pd
from collections import defaultdict
from player_hand import PlayerHand

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class Clue(object):
    def __init__(self, other_players_list, my_hand):
        self.other_players_list = other_players_list
        self.my_hand = my_hand

        rooms = ['Study', 'Kitchen', 'Hall', 'Conservatory', 'Lounge', 'Ballroom', 'Dining Room', 'Library', 'Billiard Room']
        self.remaining_rooms = [room for room in rooms if room not in self.my_hand]

        suspects = ['Plum', 'White', 'Scarlet', 'Green', 'Mustard', 'Peacock']
        self.remaining_suspects = [suspect for suspect in suspects if suspect not in self.my_hand]

        weapons = ['rope', 'dagger', 'wrench', 'pistol', 'candlestick', 'lead pipe']
        self.remaining_weapons = [weapon for weapon in weapons if weapon not in self.my_hand]

        self.turn_generator = self._turn_generator()
        self.players_hands = self._player_hands_create()

    def _turn_generator(self):
        while True:
            for player in self.other_players_list:
                yield player

    def _player_hands_create(self):
        cards_left_in_deck = self.remaining_rooms + self.remaining_suspects + self.remaining_weapons
        return {player[0]: PlayerHand(cards_left_in_deck, player[1]) for player in self.other_players_list}

    def card_reveal(self, card, player):
        """player shows me card"""
        self.players_hands[player].posterior_update('yes', card)

    def uncertain_card_reveal(self, inquiry, player, answer='no'):
        """'player' responds to the 'inquiry' with a 'no' or 'yes' by showing one of his cards. Cards are added to a
        list in either yes_dict or no_dict. If a 'no' response is given, the card values are added directly to the list.
        Otherwise, the list of cards is added to the list."""
        if answer == 'no':
            self.players_hands[player].posterior_update('no', inquiry)
        else:
            self.players_hands[player].posterior_update('yes', inquiry)

    def _cross_join(self):
        df = self.players_hands[self.other_players_list[0][0]].assign(key=1).query('posterier_prob > 0')
        for player in self.other_players_list[1:]:
            df_next = self.players_hands[player[0]].assign(key=1).query('posterier_prob > 0')
            df = df.merge(df_next, how='left', on='key')
        df.drop('key', 1, inplace=True)
        return df

    def _find_plausible_combos(self, df):
        print(df.head())

    def envelope_distribution(self):
        # todo: 1. toy hands, 2. check output, 3. plausible combos

        hand_dist_lengths = [len(self.players_hands[player[0]].possible_hands.query('posterior_prob > 0')) for player in self.other_players_list]


        print(hand_dist_lengths)
        print(functools.reduce(lambda x, y: x * y, hand_dist_lengths))
        sys.exit()
        if functools.reduce(lambda x, y: x * y, hand_dist_lengths) < 1000:
            df = self._cross_join().pipe(self._find_plausible_combos)






if __name__ == '__main__':
    players = [('Calvin', 3), ('Kay', 3), ('Martin', 3), ('Seth', 3), ('Maggie', 3)]
    my_hand = ['Study', 'Kitchen', 'Plum']
    # players = [('Calvin', 5), ('Kay', 4), ('Martin', 4)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope']
    # players = [('Calvin', 6), ('Kay', 6)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope', 'dagger']




    # todo: I'm still stuck on the best way to do this. What I have does an effective job for the players hand but doesn't
    # allow for the aggregation well.



    c = Clue(players, my_hand)
    c.card_reveal('White', 'Calvin')
    c.card_reveal('Scarlet', 'Calvin')
    c.card_reveal('Hall', 'Kay')
    c.card_reveal('Conservatory', 'Kay')
    c.card_reveal('Lounge', 'Martin')
    c.card_reveal('Ballroom', 'Martin')
    c.card_reveal('Dining Room', 'Seth')
    c.card_reveal('Green', 'Seth')
    c.card_reveal('rope', 'Maggie')
    c.card_reveal('dagger', 'Maggie')

    c.envelope_distribution()
    # print(c.players_hands['Calvin'].possible_hands)

    # rooms = ['Study', 'Kitchen', 'Hall', 'Conservatory', 'Lounge', 'Ballroom', 'Dining Room', 'Library',
    #          'Billiard Room']
    # self.remaining_rooms = [room for room in rooms if room not in self.my_hand]
    #
    # suspects = ['Plum', 'White', 'Scarlet', 'Green', 'Mustard', 'Peacock']
    # self.remaining_suspects = [suspect for suspect in suspects if suspect not in self.my_hand]
    #
    # weapons = ['rope', 'dagger', 'wrench', 'pistol', 'candlestick', 'lead pipe']


    # inquiry = ('White', 'rope', 'rope')
    # c.uncertain_card_reveal(inquiry, 'Calvin', 'no')

    # print(c.players_hands['Calvin'].possible_hands)

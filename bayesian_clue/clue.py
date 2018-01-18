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

        self.rooms = ['Study', 'Kitchen', 'Hall', 'Conservatory', 'Lounge', 'Ballroom', 'Dining Room', 'Library', 'Billiard Room']
        self.remaining_rooms = [room for room in self.rooms if room not in self.my_hand]

        self.suspects = ['Plum', 'White', 'Scarlet', 'Green', 'Mustard', 'Peacock']
        self.remaining_suspects = [suspect for suspect in self.suspects if suspect not in self.my_hand]

        self.weapons = ['rope', 'dagger', 'wrench', 'pistol', 'candlestick', 'lead pipe']
        self.remaining_weapons = [weapon for weapon in self.weapons if weapon not in self.my_hand]

        self.turn_generator = self._turn_generator()
        self.players_hands = self._player_hands_create()

    def _turn_generator(self):
        while True:
            for player in self.other_players_list:
                yield player

    def _player_hands_create(self):
        cards_left_in_deck = self.remaining_rooms + self.remaining_suspects + self.remaining_weapons
        return {player[0]: PlayerHand(cards_left_in_deck, player[0], player[1]) for player in self.other_players_list}

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

    def _find_plausible_combos(self, x):
        hand_vars = [var for var in x.index.tolist() if 'hand' in var]

        all_hand = []
        for hand in hand_vars:
            all_hand += x[hand][1:-1].replace("'", "").split(", ")

        condition1 = len(all_hand) == len(set(all_hand))  # i.e., no repeat cards in hands
        condition2 = (len([card for card in self.rooms if (card not in all_hand) and (card not in self.my_hand)]) > 0) and \
            (len([card for card in self.suspects if (card not in all_hand) and (card not in self.my_hand)]) > 0) and \
            (len([card for card in self.suspects if (card not in all_hand) and (card not in self.my_hand)]) > 0)

        if condition1 and condition2:
            x['infeasible'] = 0
            return x
        else:
            x['infeasible'] = 1
            return x


    def _cross_join(self):
        df = self.players_hands[self.other_players_list[0][0]].possible_hands.\
            assign(key=1).\
            query('posterior_prob > 0')

        for player in self.other_players_list[1:]:
            df_next = self.players_hands[player[0]].possible_hands.assign(key=1).query('posterior_prob > 0')
            df = df.\
                merge(df_next, how='left', on='key').\
                apply(self._find_plausible_combos, axis=1). \
                query('infeasible == 0'). \
                drop('infeasible', 1)
        df.drop('key', 1, inplace=True)
        return df

    def envelope_distribution(self):
        print(self._cross_join())


# todo: finish the enevelope df creation
#     -should display possible envelope as well
# todo: make displaying a players possible hand easy
# todo: add the calculate cardinality of possible hands to the end of the card_reveal and uncertain_card_reveal methods
# todo: the posterior probability has to be .5 not .062
#     -the problem is I cannot (it's too big) calculate this every time and so getting a true conditional posterior is not possible.


# 1. is there anything that can be inferred from combining all hands into 1 dataset? No, other than probabilities for hands.
# 2. just wait until there is only one possible envelope
# 3. how do you reduce the number of rows in a posible_hand df? I have to ask good questions.



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
    c.card_reveal('wrench', 'Calvin')
    c.card_reveal('Hall', 'Kay')
    c.card_reveal('Conservatory', 'Kay')
    c.card_reveal('Peacock', 'Kay')
    c.card_reveal('Lounge', 'Martin')
    c.card_reveal('Ballroom', 'Martin')
    c.card_reveal('Library', 'Martin')
    c.card_reveal('Dining Room', 'Seth')
    c.card_reveal('Green', 'Seth')
    c.card_reveal('pistol', 'Seth')
    c.card_reveal('rope', 'Maggie')
    c.card_reveal('dagger', 'Maggie')
    # c.card_reveal('candlestick', 'Maggie')

    c.envelope_distribution()
    # print(c.players_hands['Calvin'].possible_hands)

    #          'Billiard Room', 'Mustard, 'lead pipe'


    # inquiry = ('White', 'rope', 'rope')
    # c.uncertain_card_reveal(inquiry, 'Calvin', 'no')

    # print(c.players_hands['Calvin'].possible_hands)

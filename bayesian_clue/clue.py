import sys
import itertools
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


#     todo: need something that combines together the information in each of the individual hands

if __name__ == '__main__':
    players = [('Calvin', 3), ('Kay', 3), ('Martin', 3), ('Seth', 3), ('Maggie', 3)]
    my_hand = ['Study', 'Kitchen', 'Plum']
    # players = [('Calvin', 5), ('Kay', 4), ('Martin', 4)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope']
    # players = [('Calvin', 6), ('Kay', 6)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope', 'dagger']

    c = Clue(players, my_hand)
    # c.card_reveal('White', 'Calvin')
    # print(c.players_hands['Calvin'].possible_hands)

    # inquiry = ('White', 'rope', 'rope')
    # c.uncertain_card_reveal(inquiry, 'Calvin', 'no')

    # print(c.players_hands['Calvin'].possible_hands)

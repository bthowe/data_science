import sys
import itertools
import pandas as pd
from collections import defaultdict

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

        self.player_generator = self._player_generator()
        self.possible_hands = self._possible_hands()

    def _player_generator(self):
        while True:
            for player in self.other_players_list:
                yield player

    def _possible_hands(self):
        player_hand = []
        # 112
        # 924 for 3 players, ~84k for 4, ~something huge for 6 (just think, 18 players would be 17!)
        for envelope in itertools.product(self.remaining_rooms, self.remaining_suspects, self.remaining_weapons):
            remaining_cards = [card for card in self.remaining_rooms + self.remaining_suspects + self.remaining_weapons if card not in envelope]

            for hand in self._hands_create(remaining_cards):
                player_hand.append([str(envelope)] + hand)

            # columns = ['envelope'] + [players[0] for players in self.other_players_list]
            # print(pd.DataFrame(player_hand, columns=columns).head(50))
            # sys.exit()
            print(len(player_hand))
            sys.exit()
        columns = ['envelope'] + [players[0] for players in self.other_players_list]
        return pd.DataFrame(player_hand, columns=columns)

    def _hands_create(self, cards):
        hands_lst = []

        last_player_card_count = self.other_players_list[-1][1]
        while len(cards) > last_player_card_count:
            next_player = next(self.player_generator)[1]
            for player_hand in itertools.combinations(cards, next_player):
                remaining_cards = [card for card in cards if card not in player_hand]
                hand_combination = self._hands_create(remaining_cards)
                if type(hand_combination) == tuple:
                    hands_lst.append([str(player_hand), str(hand_combination)])
                else:
                    for hand in hand_combination:
                        hands_lst.append([str(player_hand)] + hand)
            return hands_lst
        return tuple(cards)

    def hands_update(self, player, cards, answer='no'):
        pass

    def card_reveal(self, player, card):
        pass

    def turn(self, inquiry, player, answer='no'):
        self.history.append([inquiry, player, answer])

    def card_reveal(self, player, card):

        pass

# what if I had a hand of potential cards for each of the players and as they say no I widdle each down. Then I somehow use this to construct the possibilities
# I should calculate the number of possibly hands as well and when it gets down to something reasonable, create the dataframe
# It should be pretty straightforward to calculate the possible number of hands
# Then I'd update the dataframe using the historical and future data

# players = 2
# cards_num = 2
# def hand1(h):
#     player_hand = []
#     for envelope in itertools.combinations(h, 3):
#         h_minus_envelope = [card for card in h if card not in envelope]
#         for hand in hand2(envelope, h_minus_envelope):
#             player_hand.append([str(envelope)] + hand)
#
#     return player_hand
#
# def hand2(e, h):
#     player_hand_lst = []
#     while len(h) > cards_num:
#         for player_hand in itertools.combinations(h, 2):
#             h_minus_envelope = [card for card in h if card not in player_hand]
#             h2 = hand2(e, h_minus_envelope)
#             if type(h2) == tuple:
#                 player_hand_lst.append([str(player_hand), str(h2)])
#             else:
#                 for hand in h2:
#                     player_hand_lst.append([str(player_hand)] + hand)
#         return player_hand_lst
#     return tuple(h)

# todo: convert to pandas dataframe
# todo: clean up this nasty code


if __name__ == '__main__':
    players = [('Calvin', 3), ('Kay', 3), ('Martin', 3), ('Seth', 3), ('Maggie', 3)]
    my_hand = ['Study', 'Kitchen', 'Plum']
    # players = [('Calvin', 5), ('Kay', 4), ('Martin', 4)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope']
    # players = [('Calvin', 6), ('Kay', 6)]
    # my_hand = ['Study', 'Kitchen', 'Plum', 'White', 'rope', 'dagger']

    c = Clue(players, my_hand)
    # print(c.possible_hands.head(50))
    print(c.possible_hands.shape)
    print(c.possible_hands.drop_duplicates(keep='last').shape)
    # maybe I'll have to wait a couple of rounds to restrict the size

    # h = range(9)
    # hand = hand1(h)
    # print(pd.DataFrame(hand, columns=['envelope', 'player1', 'player2', 'player3']).head(50))

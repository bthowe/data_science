import sys
import itertools
import pandas as pd
from collections import defaultdict


class Clue(object):
    def __init__(self, other_players_dict, my_hand):
        self.other_players_dict = other_players_dict
        self.other_players = other_players_dict.keys()  # making a list of player names since ordered
        self.my_hand = my_hand
        self.rooms = [
            'Study',
            'Kitchen',
            'Hall',
            'Conservatory',
            'Lounge',
            'Ballroom',
            'Dining Room',
            'Library',
            'Billiard Room'
        ]
        self.suspects = [
            'Plum',
            'White',
            'Scarlet',
            'Green',
            'Mustard',
            'Peacock'
        ]
        self.weapons = [
            'rope',
            'dagger',
            'wrench',
            'pistol',
            'candlestick',
            'lead pipe'
        ]
        self._possible_hands()

    def _possible_hands(self):
        remaining_rooms = [room for room in self.rooms if room not in self.my_hand]
        remaining_suspects = [suspect for suspect in self.suspects if suspect not in self.my_hand]
        remaining_weapons = [weapon for weapon in self.weapons if weapon not in self.my_hand]

        for i in itertools.product(remaining_rooms, remaining_suspects, remaining_weapons):
            envelope = i
            print(envelope)
            remaining_cards = [room for room in remaining_rooms if room not in envelope] + \
                              [suspect for suspect in remaining_suspects if suspect not in envelope] + \
                              [weapon for weapon in remaining_weapons if weapon not in envelope]
            print(remaining_cards)
            print(self._shuffle(remaining_cards))
            sys.exit()
            print(self._deal(self._shuffle(remaining_cards)))

    def _shuffle(self, deck):
        # todo: take all combinations of true/false instead...what if there are more than two other players?
        # todo: instead of using True and False, use integers for each player

        player_cards = []
        for player in self.other_players:
            player_cards += [player] * self.other_players_dict[player]

        df_hands = pd.DataFrame(list(itertools.permutations(player_cards)), columns=deck)
        print(df_hands.head())
        print(df_hands.shape)
        sys.exit()

        # return itertools.permutations(deck)

    def _deal(self, deck):
        deck_lst = list(deck)
        player_hand_dic = {player:[] for player in self.other_players}
        pool = itertools.cycle(self.other_players)
        while deck_lst:
            player_hand_dic[next(pool)].append(deck_lst.pop())
        return player_hand_dic

    def hands_update(self, player, cards, answer='no'):
        pass

    def card_reveal(self, player, card):





if __name__ == '__main__':
    players = {'Calvin': 6, 'Kay': 6}
    my_hand = ['Study', 'Kitchen', 'Hall', 'Plum', 'White', 'rope', 'dagger']

    c = Clue(players, my_hand)

    # todo: permutations will lead to duplicate hands

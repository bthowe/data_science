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

        for other_player in self.other_players_list:
            if other_player[0] != player:
                self.players_hands[other_player[0]].posterior_update('no', [card])

    def uncertain_card_reveal(self, inquiry, player, answer='no'):
        """'player' responds to the 'inquiry' with a 'no' or 'yes' by showing one of his cards. Cards are added to a
        list in either yes_dict or no_dict. If a 'no' response is given, the card values are added directly to the list.
        Otherwise, the list of cards is added to the list."""
        if answer == 'no':
            self.players_hands[player].posterior_update('no', inquiry)
        else:
            self.players_hands[player].posterior_update('yes', inquiry)
            # todo: if someone says yes, then all other players are no for hands including all of those three cards.


    def _lister(self, x):
        return x[1:-1].replace("'", "").split(", ")

    def _find_plausible_combos(self, x):
        hand_vars = [var for var in x.index.tolist() if 'hand' in var]

        all_hand = []
        for hand in hand_vars:
            all_hand += self._lister(x[hand])

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
        df = pd.DataFrame([[str(tuple(self.my_hand))]], columns=['hand_I']).\
            assign(key=1)

        for player in self.other_players_list:
            df_next = self.players_hands[player[0]].possible_hands[['hand_{}'.format(player[0]), 'posterior_prob']].assign(key=1).query('posterior_prob > 0')
            df = df.\
                merge(df_next.drop('posterior_prob', 1), how='left', on='key').\
                apply(self._find_plausible_combos, axis=1). \
                query('infeasible == 0'). \
                drop('infeasible', 1)
        df.drop('key', 1, inplace=True)
        return df

    def _envelope_create(self, x):
        hands_all = []
        for val in x.iteritems():
            hands_all += self._lister(val[1])
        x['envelop'] = [card for card in self.rooms + self.weapons + self.suspects if card not in hands_all]
        return x

    def possible_envelope(self):
        print(self._cross_join().apply(self._envelope_create, axis=1))

    def dist_of_possible_cards(self):
        for player in self.other_players_list:
            print(self.players_hands[player[0]].possible_hands.query('posterior_prob > 0'))


if __name__ == '__main__':
    players = [('Calvin', 3), ('Kay', 3), ('Martin', 3), ('Seth', 3), ('Maggie', 3)]
    my_hand = ['Study', 'Kitchen', 'Plum']

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

    c.dist_of_possible_cards()
    # c.possible_envelope()

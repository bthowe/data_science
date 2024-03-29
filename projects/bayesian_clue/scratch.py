import functools

def _cross_join():
    df = self.players_hands[self.other_players_list[0][0]].assign(key=1).query('posterier_prob > 0')
    for player in self.other_players_list[1:]:
        df_next = self.players_hands[player[0]].assign(key=1).query('posterier_prob > 0')
        df = df.merge(df_next, how='left', on='key')
    df.drop('key', 1, inplace=True)
    return df


def _find_plausible_combos(df):
    print(df.head())


def envelope_distribution(hand_dict):
    # todo: 1. toy hands, 2. check output, 3. plausible combos

    hand_dist_lengths = [len(self.players_hands[player[0]].query('posterior_prob > 0')) for player in
                         self.other_players_list]
    if functools.reduce(lambda x, y: x * y, hand_dist_lengths) < 1000:
        df = self._cross_join().pipe(self._find_plausible_combos)


if __name__ == '__main__':
    hands = {'Calvin': '(a, b, c)', 'Kay': '(d, e, f'}


    envelope_distribution()



# # players = 2
# # cards_num = 2
# # def hand1(h):
# #     player_hand = []
# #     for envelope in itertools.combinations(h, 3):
# #         h_minus_envelope = [card for card in h if card not in envelope]
# #         for hand in hand2(envelope, h_minus_envelope):
# #             player_hand.append([str(envelope)] + hand)
# #
# #     return player_hand
# #
# # def hand2(e, h):
# #     player_hand_lst = []
# #     while len(h) > cards_num:
# #         for player_hand in itertools.combinations(h, 2):
# #             h_minus_envelope = [card for card in h if card not in player_hand]
# #             h2 = hand2(e, h_minus_envelope)
# #             if type(h2) == tuple:
# #                 player_hand_lst.append([str(player_hand), str(h2)])
# #             else:
# #                 for hand in h2:
# #                     player_hand_lst.append([str(player_hand)] + hand)
# #         return player_hand_lst
# #     return tuple(h)
#
# # todo: convert to pandas dataframe
# # todo: clean up this nasty code
#
#
#
#
#
#
#
#
#
#
#     def _possible_hands(self):
#         player_hand = []
#         # 112
#         # 924 for 3 players, ~84k for 4, ~something huge for 6 (just think, 18 players would be 17!)
#         for envelope in itertools.product(self.remaining_rooms, self.remaining_suspects, self.remaining_weapons):
#             remaining_cards = [card for card in self.remaining_rooms + self.remaining_suspects + self.remaining_weapons if card not in envelope]
#
#             for hand in self._hands_create(remaining_cards):
#                 player_hand.append([str(envelope)] + hand)
#
#             # columns = ['envelope'] + [players[0] for players in self.other_players_list]
#             # print(pd.DataFrame(player_hand, columns=columns).head(50))
#             # sys.exit()
#             print(len(player_hand))
#             sys.exit()
#         columns = ['envelope'] + [players[0] for players in self.other_players_list]
#         return pd.DataFrame(player_hand, columns=columns)
#
#     def _hands_create(self, cards):
#         hands_lst = []
#
#         last_player_card_count = self.other_players_list[-1][1]
#         while len(cards) > last_player_card_count:
#             next_player = next(self.player_generator)[1]
#             for player_hand in itertools.combinations(cards, next_player):
#                 remaining_cards = [card for card in cards if card not in player_hand]
#                 hand_combination = self._hands_create(remaining_cards)
#                 if type(hand_combination) == tuple:
#                     hands_lst.append([str(player_hand), str(hand_combination)])
#                 else:
#                     for hand in hand_combination:
#                         hands_lst.append([str(player_hand)] + hand)
#             return hands_lst
#         return tuple(cards)
#
#     def hands_update(self, player, cards, answer='no'):
#         pass
#
#
#     self.yes_dict = {player[0]: [] for player in self.other_players_list}
#     self.yes_uncertain_dict = {player[0]: [] for player in self.other_players_list}
#     self.no_dict = {player[0]: [] for player in self.other_players_list}
#     def card_reveal(self, player, card):
#         """player shows me card"""
#         self.yes_dict[player] += card
#
#     def turn(self, inquiry, player, answer='no'):
#         """'player' responds to the 'inquiry' with a 'no' or 'yes' by showing one of his cards. Cards are added to a
#         list in either yes_dict or no_dict. If a 'no' response is given, the card values are added directly to the list.
#         Otherwise, the list of cards is added to the list."""
#         if answer == 'no':
#             self.no_dict[player] += inquiry
#         else:
#             self.yes_uncertain_dict[player].append(inquiry)
#
#
#     def players_hand_create(self):
#
#         self.remaining_rooms + self.remaining_suspects + self.remaining_weapons
#
# # why don't I simply, for each player, (18 choose whatever).
# # for each player, create a player hand class.
#
#
# # what if I had a hand of potential cards for each of the players and as they say no I widdle each down. Then I somehow use this to construct the possibilities
# # I should calculate the number of possibly hands as well and when it gets down to something reasonable, create the dataframe
# # It should be pretty straightforward to calculate the possible number of hands
# # Then I'd update the dataframe using the historical and future data
# # don't be explicit about possible envelopes...I can back that out.
#

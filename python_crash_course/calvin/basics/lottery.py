lottery_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'a', 'b', 'c', 'd', 'e']

from random import choice
winner_1 = choice(lottery_numbers)
winner_2 = choice(lottery_numbers)
winner_3 = choice(lottery_numbers)
winner_4 = choice(lottery_numbers)

print(f"whoever got one of these numbers wins: {winner_1}, {winner_2}, {winner_3}, {winner_4}")

my_ticket = choice(lottery_numbers)
print(f"Your ticket is {my_ticket}")

# for ticket in my_ticket:
if my_ticket == winner_1:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_2:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_3:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_4:
    print('YOU WON WINNING TICKET!')
else:
    print('You have no winning ticket.')




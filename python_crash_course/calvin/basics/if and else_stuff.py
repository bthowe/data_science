best = 'Paladin'
print("Is best == 'Paladin'.I predict True." )
print(best == 'Paladin')
print("\nIs best == 'Warlock'? I don't think so.")
print(best == 'Warlock')
print("\nIs best =='Druid'? I don't think so.")
print(best == 'Druid')

best_ranged_weapon = 'longbow'
print("\nIs best ranged weapon == 'heavy crossbow'? No")
print(best_ranged_weapon == 'heavy crossbow')
print("\nIs best ranged weapon == 'longbow'? Yes")
print(best_ranged_weapon == 'longbow')
print("\nIs best ranged weapon == 'shortbow'? No")
print(best_ranged_weapon == 'shortbow')

good_book = "Players Handbook"
print("\nIs a good book the'Players Handbook'? Yes")
print(good_book == 'Players Handbook')
print("\nIs food book == the 'Hunger Games'? No")
print(good_book == 'Hunger Games')

better_book = 'The Way of Kings'
print("\nIs better book == 'The Way of Kings'? Yes")
print(better_book == 'The Way of Kings')

worst_book = 'The Personel Memoirs of U.S.Grant'
print("\nIs the worst book 'The Personel Memoirs of U.S.Grant'? Yes")
print(worst_book == 'The Personel Memoirs of U.S.Grant')


best_class = ['Paladin','Warlock','Druid','Rogue']
worst = 'Sorcerer'

if worst not in best_class:
    print(f"\nI'm not going to use a {worst} character.")
    print(worst == best_class)

'Warlock' in best_class
'Sorcerer' in best_class


car = 'Toyota'
print(car == 'toyota')

print(car.lower() == 'toyota')

# age = 18
# print(age == 21)
# print(age == 18)
# print(age <= 30)
# print(age >= 15)
# print(age >= 30)
# print(age <= 15)

age_0 = 23
age_1 = 83
age_2 = 12

print(age_0 >= 21 and age_1 >= 21)

print(age_0 <= 24 or age_1 >= 89)

worst_weapons = ('club','net','whip','sling')
print('dagger' in worst_weapons)
print('club' in worst_weapons)

best_weapons = ['greatsword','longbow','battle axe']
if best_weapons != 'missle':
    print("Maybe they only had one missle.")


alien_color = 'green'

if 'green' in alien_color:
    print('You earned five points.')

if 'red' in alien_color:
    print('You earn fifteen points.')

alien_colors = 'yellow'

if 'green' in alien_colors:
    print('You just earned five points.')
else:
    print('You earned ten points.')

if 'green' in alien_colors:
    print('You just earned five points.')
elif 'yellow' in alien_colors:
    print('You just earned ten points.')
else:
    print('You just earned fifteen points.')

if 'red' in alien_colors:
    print('You just earned fifteen points.')
elif 'green' in alien_colors:
    print('You just earned five points.')
else:
    print('You just earned ten points.')

if 'yellow' in alien_colors:
    print('You just earned ten points.')
elif 'red' in alien_colors:
    print('you just earned fifteen points.')
else:
    print('you just earned five points.')

age = 45

if age < 2:
    print('You are a baby.')
elif 4 > age >= 2:
    print('You are a toddler.')
elif 13 > age >= 4:
    print('You are a kid.')
elif 20 > age >= 13:
    print('You are a teenager.')
elif 65 > age >= 20:
    print('You are an adult.')
else:
    print('You are a senior.')

favorite_food = ['raspberry','pomegranite','cherries']

if 'raspberry' in favorite_food:
    print('You love ripe raspberries.')
if 'banana' in favorite_food:
    print("You don't like ripe nanners.")
if 'pomegranite' in favorite_food:
    print('Pomegranites are amazing.')
if 'strawberries' in favorite_food:
    print('Strawberries are good.')
if 'cherries' in favorite_food:
    print('You like cherries.')


user_names = []
if user_names:
    for user_name in user_names:
        if user_name == 'admin':
            print('Hello admin, would you like to see a status report?')
    else:
        print(f"Hello {user_name}, thank you for logging in.")
else:
    print('We need to find some users!')

current_users = ['George','Sticky','Mallory','Frodo','Perrin']
new_users = ['Kate','Frodo','Perrin','Godfrey','Mat']

for users in new_users:
    if users in current_users:
        print("You need a new user name.")
    else:
        print('This user name is avalible.')


numbers = [1,2,3,4,5,6,7,8,9]
for num in numbers:
    if num is 1:
        print('1st')
    elif num is 2:
        print('2nd')
    elif num is 3:
        print('3rd')
    else:
        print(f"{num}th")
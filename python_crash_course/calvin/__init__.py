class Restaurant:
    def __init__(self, name, cuisine,):
        self.name = name
        self.cuisine = cuisine

    """Print the name of this restaurant"""
    def place_name(self):
        print(f"This restaurant is named {self.name}.")

    """Print what kind of food this restaurant has"""
    def food(self):
        print(f"This restaurant has {self.cuisine} food.")

describe_restaurant = Restaurant('Red Robin','chicken fingers')

describe_restaurant.place_name()
describe_restaurant.food()

print(describe_restaurant.place_name())
print(describe_restaurant.food())

least_favorite_restaurant = Restaurant('macdonalds','burgers')
favorite_restaurant = Restaurant("wendy's",'burgers')
good_restaurant = Restaurant('chickfillay','chicken')

least_favorite_restaurant.place_name()
least_favorite_restaurant.food()

favorite_restaurant.place_name()
favorite_restaurant.food()

good_restaurant.place_name()
good_restaurant.food()



class User:

    def __init__(self, first_name, last_name, age, height, richness):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.height = height
        self.richness = richness

    """Print a summery of this information"""
    def describe_user(self):
        print(f"\n{self.first_name} {self.last_name} is {self.age} years old,"
              f" {self.height} tall,"
              f" and {self.richness} rich")

    def greet_user(self):
        print(f"Hello {self.first_name} {self.last_name}.")


me = User('Calvin', 'Howe', '12', "5'3", '$700')

me.describe_user()
me.greet_user()

samuel = User('Samuel', 'Howe', '10', 'unknown', "'round $800")

samuel.describe_user()
samuel.greet_user()


class Restaurants:
    def __init__(self, name, cuisine,number_served):
        self.name = name
        self.cuisine = cuisine
        self.number_served = number_served
        self.number_served = 0
    """Print the name of this restaurant"""
    def place_name(self):
        print(f"This restaurant is named {self.name}.")

    """Print what kind of food this restaurant has"""
    def food(self):
        print(f"This restaurant has {self.cuisine} food.")

    """Print how many customers have been served"""
    def set_number_served(self, people):
        self.number_served = people
        print(f"This restaurant has served {self.number_served} people.")

    def increment_number_served(self, number):
        self.number = number
        self.number_served += number
        print(f"{self.number}")



restaurant = Restaurants('Red Lobster','seafood',0)
restaurant.food()
restaurant.place_name()
restaurant.set_number_served(people=0)
restaurant.increment_number_served(number=99999999999999999999999999999999999999999999999999999999999999999999999999999)
print(f"{restaurant.increment_number_served(number=99999999999999999999999999999999999999999999999999999999999999999)}")

class Users:
    def __init__(self, first_name, last_name, age, height, richness,):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.height = height
        self.richness = richness
        self.login_attempts = 0

    """Print a summery of this information"""
    def describe_user(self):
        print(f"\n{self.first_name} {self.last_name} is {self.age} years old, {self.height} tall, and {self.richness} rich")

    def greet_user(self):
        print(f"Hello {self.first_name} {self.last_name}.")

    def increment_login_attempts(self, login):
        self.login_attempts += login
        print(f"You have tried logging in {login}.")

you = Users('Hou','Tenderfoot',89,"11'7",'100100100')

you.describe_user()
you.greet_user()
you.increment_login_attempts(2)


class IceCreamStand(Restaurant):

    def __init__(self, name, cuisine, flavors):
        super().__init__(cuisine, flavors)
        self.flavors = flavors

    def ice_cream(self):
        self.flavors = ['vanilla','cookie doe','chocolate','moose tracks','strawberry','chunky tofu','rainbow sparkle',
                        'butterscotch','pecon','sea salt caramel','raspberry','mint','pistachio']
        print(f"These are the ice cream flavors: {self.flavors}")

my_dinner = IceCreamStand('vanilla',"culver's", "burger",)

my_dinner.place_name()
my_dinner.food()
my_dinner.ice_cream()


class Privileges:

    def __init__(self, privileges):
        self.privileges = privileges

    def show_privileges(self):
        self.privileges = ['can delete posts', 'can add posts', 'can ban users', 'can appoint deputy admins']
        print(f"I, Admin, {self.privileges}")

class Admin(User):

    def __init__(self, first_name, last_name, age, height, richness, privilege):
        super().__init__(first_name, last_name, age, height, richness)
        self.privileges = Privileges(privilege)


admin = Admin('Admin', 'Admin', 123, "5'6", '$800000000','stuff')

admin.greet_user()
admin.describe_user()
admin.privileges.show_privileges()

mr_admin = Privileges




class Car:

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name.title()

    def read_odometer(self):
        print(f"This car has {self.odometer_reading} miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles

#class ElectricCar(Car):

#    def __init__(self, make, model, year, battery):

#        super().__init__(make, model, year)
#        self.battery = Battery(battery)

#my_ecar = ElectricCar('ferrari', 'best model', 1356, 75)
#my_ecar.battery.get_range()
#my_ecar.battery.describe_battery()
#my_ecar.battery.upgrade_battery()

#class Battery:

#    def __init__(self, battery_size=75):
#        self.battery_size = battery_size
#        # self.battery_size = 75
#
#    def describe_battery(self):
#        print(f"This car has a {self.battery_size}-kWh battery.")

#    def upgrade_battery(self):
#        if self.battery_size <= 100:
#            self.battery_size = 100


#    def get_range(self):
#        if self.battery_size == 75:
#            range = 260
#        elif self.battery_size == 100:
#            range = 315

#        print(f"\nThis car can go about {range} miles on a full charge.")


#class ElectricCar(Car):

#    def __init__(self, make, model, year, battery):

#        super().__init__(make, model, year)
#        self.battery = Battery(battery)

#my_ecar = ElectricCar('ferrari', 'best model', 1356, 75)
#my_ecar.battery.get_range()
#my_ecar.battery.describe_battery()
#my_ecar.battery.upgrade_battery()
#print(my_ecar.battery.battery_size)

from random import randint

class Die:

    def __init__(self, sides):
        self.sides = sides

    def d_9(self):
        self.sides = 9
        print(randint(1, self.sides))

    def d_12(self):
        self.sides = 12
        print(randint(1, self.sides))

    def d_20(self):
        self.sides = 20
        print(randint(1, self.sides))

    def d_50(self):
        self.sides = 50
        print(randint(1, self.sides))

    def d_100(self):
        self.sides = 100
        print(randint(1, self.sides))

    def d_150(self):
        self.sides = 150
        print(randint(1, self.sides))




    def roll_dice(self):
        self.sides = 1000
        print(randint(1, self.sides))

my_die = Die(10000000)
my_die.roll_dice()
my_die.d_9()
my_die.d_12()
my_die.d_20()
my_die.d_50()
my_die.d_100()
my_die.d_150()


lottery_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'a', 'b', 'c', 'd', 'e']

from random import choice
winner_1 = choice(lottery_numbers)
winner_2 = choice(lottery_numbers)
winner_3 = choice(lottery_numbers)
winner_4 = choice(lottery_numbers)

print(f"Whoever got one of these numbers wins: {winner_1}, {winner_2}, {winner_3}, {winner_4}")

my_ticket = choice(lottery_numbers)
print(f"Your ticket is number {my_ticket}")

if my_ticket == winner_1:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_2:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_3:
    print('YOU WON WINNING TICKET!')
elif my_ticket == winner_4:
    print('YOU WON WINNING TICKET!')
else:
    print('You have won no winning ticket.')


filename = 'basics/learning_python.txt'

with open(filename) as file_object:
    for line in file_object:
        print(line.replace('C', 'Python').rstrip())


guest = input('What is your name? ')

filename = 'basics/guest.txt'

with open(filename) as file_object:

    for person in file_object:
        person = guest
        print(person)

guests_live = True

name = input('What is your name Mr. Guest? ')
print(f'We are currently full Mr. {name}, but if you brought a tent you can sleep in the parking lot.')

while guests_live:
    if name == 'Room Service':
        break
    elif name == 'stop':
        guests_live = False

    else:
        filename = 'basics/guest_book.txt'
        with open(filename) as file_object:

            for guest in file_object:
                guest = name




poll = input('Why are you learning Python? ')
programmers_program = True

while programmers_program:
    filename = 'basics/poll_results.txt'
    with open(filename) as file_object:

        for polled in file_object:
            polled = poll

        if poll == 'stop':
            programmers_program = False



print("Give me two numbers and I will multiply them")
print("Enter q to quit")

while True:
    first_number = input("\nFirst number: ")
    if first_number == 'q':
        break
    second_number = input("\nSecond number: ")
    if second_number == 'q':
        break
    try:
        answer = int(first_number) * int(second_number)
    except ValueError:
        print('Give the numbers in numerical form. You cannot use words')
    else:
        print(answer)


class Employee:

    def __init__(self, first, last, salary, extra):
        self.first = first
        self.last = last
        self.salary = salary
        self.extra = extra(5000)

    def give_raise(self):
        print(f"{self.first} {self.last}, salary {self.salary}, going to get a pay raise of {self.extra}")


me = Employee('Calvin', 'Howe', '$6')
me.give_raise()


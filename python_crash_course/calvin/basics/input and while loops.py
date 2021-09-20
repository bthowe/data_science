


dinner_goers = input("How many are in your group?")
dinner_goers = int(dinner_goers)

if dinner_goers > 8:
    print("\nYou'l have to wait for a table.")
else:
    print("\nYour table is ready.")


number = input("Tell me a number and i'll tell you if it's a multiplier of ten.")
number = int(number)

if number % 10 == 0:
   print(f"{number} is a multiplier of ten")
else:
   print(f"{number} is not a multiplier of ten.")


prompt = "\nPlease enter what pizza toppings you would like."
prompt += "\nEnter quit when you are done."

while True:
    topping = input(prompt)

    if topping == 'quit':
        break
    else:
        print(f"\nAdding {topping}.")


ask_age = "What age are you? "

while True:
    question = int(input(ask_age))
    if question == 999:
        break
    elif 0 <= question <= 3:
        print("\nYour ticket is free.")
    elif 3 < question <= 12:
        print("\nYour ticket is $10")
    else:
        print("\nYour ticket is $15.")


"""This loop runs forever!"""
a = 1
while a >= 5:
    print(a)
    a += 18


sandwich_orders = ['pastrami','peanut butter','pastrami','tuna','pastrami','honey']

finished_sandwiches = []

while 'pastrami' in sandwich_orders:
        sandwich_orders.remove('pastrami')

while sandwich_orders:
    made_sandwiches = sandwich_orders.pop()
    print("\nWe are out of pastrami.")
    print(f"\nYour {made_sandwiches} is completed.")
    finished_sandwiches.append(made_sandwiches)

print("\nI made these sandwiches.")
for sandwich in finished_sandwiches:
    print(sandwich.title())


polling_active = True

places = {}



while polling_active:
    name = input("\nWhat is your name? ")
    place = input("If you could go anywhere in the world, where would you go? ")

    places[name] = place

    repeat = input("Would you let another person respond? (yes/no)")
    if repeat == 'no':
        polling_active = False
print("\n--- Poll Results ---")
for name, place in places.items():
    print(f"{name} would like to go to {place}\n")


poll = input('Why are you learning Python? ')
programmers_program = True

while programmers_program:
    filename = 'poll_results.txt'
    with open(filename) as file_object:

        for polled in file_object:
            polled = poll
            print(polled)

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



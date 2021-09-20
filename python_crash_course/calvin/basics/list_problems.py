names = ['The Way of Kings','Lord of the Rings','The Eye of the Wourld']
print(names[0])
print(names[1])
print(names[2])

message = f"{names[0]} is a great book."
print(message)

new_message = f"{names[1]} is epic fantasy"
print(new_message)

newest_message = f"{names[2]} is really long"
print(newest_message)

big_message = f"{names[0]} is as good as the classics {names[2]}, and {names[1]}."
print(big_message)

modes_of_travel = ['walking','cycling','riding']

walk = f"I like {modes_of_travel[0]}."
print(walk)

bikes = f"{modes_of_travel[1].title()} is super fun."
print(bikes)

horse = f"I haven't done much {modes_of_travel[-1]}."
print(horse)

dinner_people = ['Grandmother','Grandpa','Beowulf']
dinner_invatation = f"You are invited to dinner {dinner_people[0]}."
dinner_invatation2 = f"You are invited to dinner {dinner_people[1]}."
dinner_guy3 = f"You are invited to dinner {dinner_people[2]}."
print(dinner_invatation)
print(dinner_invatation2)
print(dinner_guy3)

print(dinner_people[2])

del dinner_people[2]
print(dinner_people)

dinner_people.insert(2,'Webster')
print(f"You are invited to dinner {dinner_people[0]}.")

print('I found a bigger dinner table! Everyone can come.')

dinner_people.insert(3,"Rand al'thor")
dinner_people.insert(2,'Monty Python')
dinner_people.insert(0,'Frodo Baggins')
print(dinner_people)

print(f"You are invited to dinner, {dinner_people[5]}.")
print(f"You are invited to dinner, {dinner_people[4]}.")
print(f"You are invited to dinner, {dinner_people[3]}.")
print(f"You are invited to dinner, {dinner_people[2]}.")
print(f"You are invited to dinner, {dinner_people[1]}.")
print(f"You are invited to dinner, {dinner_people[0]}.")


print('I can only have space for two guests. Sorry.')

print(dinner_people)

guest = dinner_people.pop(0)
print(f"Sorry {guest}, you cannot come over for dinner. ")

guest1 = dinner_people.pop(2)
print(f"Sorry {guest1}, you cannot come over for dinner. ")

guest2 = dinner_people.pop(2)
print(f"Sorry {guest2}, you cannot come over for dinner. ")

guest3 = dinner_people.pop(2)
print(f"Sorry {guest3}, you cannot come over for dinner. ")

print(f"{dinner_people[0]}, you are still invited.")
print(f"{dinner_people[1]}, you are still invited.")

del dinner_people[0]
print(dinner_people)

del dinner_people[0]
print(dinner_people)

places_to_go = ['Rome','Giza','Athens','the Yukon','London']
print(places_to_go)

print(sorted(places_to_go))

print(places_to_go)

places_to_go.reverse()
print(places_to_go)

places_to_go.reverse()
print(places_to_go)

places_to_go.sort()
print(places_to_go)

places_to_go.sort(reverse=True)
print(places_to_go)

print(len(dinner_people))

best_classes = ['Paladin','Warlock','Rogue','Druid']

print(best_classes[0])

print(best_classes[-1])

new_class = f"{best_classes[2]}s are great."
print(new_class)

best_classes[0] = 'Fighter'
print(best_classes)

best_classes.append('Paladin')
print(best_classes)

best_classes.insert(0, 'Cleric')
print(best_classes)

del best_classes[1]
print(best_classes)

popped_best_classes = best_classes.pop()
print(best_classes)
print(popped_best_classes)

worst = best_classes.pop(0)
print(f"The best supporter class are {worst}s. ")

best_classes.remove('Druid')
print(best_classes)

best_classes.sort()
print(best_classes)

best_classes.sort(reverse=True)
print(best_classes)


best_classes.insert(0,'Barbarian')
best_classes.insert(1,'Bard')
best_classes.insert(2,'Wizard')
print(best_classes)

print(sorted(best_classes))

best_classes.reverse()
print(best_classes)

print(len(best_classes))

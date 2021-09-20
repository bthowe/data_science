person = {'first_name': 'Calvin','last_name': 'Howe','age': 12,'city': 'Kansas City'}

print(person['first_name'])
print(person['last_name'])
print(person['age'])
print(person['city'])


numbers = {'Calvin': [20,12,6,8,10],'geoffrey': [2,0],'lily': [12],'john': [13,100,20],'brandon': [10]}

print(f"\nCalvin's favorite numbers are {numbers['Calvin']}.")
print(f"\ngeoffrey's favorite numbers are {numbers['geoffrey']}.")
print(f"\nlily's favorite numbers are {numbers['lily']}.")
print(f"\njohn's favorite numbers  are {numbers['john']}.")
print(f"\nbrandon's favorite numbers are {numbers['brandon']}.")


py_words = {'print': 'print shows the output.','for loop':
    'for loops loop through the values of a list.','list':
    'A list is a group of data.','variables':
    'A variable represents a piece of data.','python':
    'python is a type of programing.','dictionary':
    'A dictionary is another way of storing data.','del':
    'del deletes elements of a list.','insert':
    'insert puts values into a list.','title':
    'title is how you make things uppercase.', 'comment':
    'A comment lets you write notes in English'}


for key, value in py_words.items():
    print(f"\nWord: {key}")
    print(f"Meaning: {value}")

rivers_of_the_world = {'Nile': 'Egypt','Amazon': 'Brazil','Mississipie': 'USA'}
for k in rivers_of_the_world:
    if k == 'Nile':
        print('\nThe Nile runs through Egypt')
    if k == 'Amazon':
        print('\nThe Amazon might be the longest river in the world.')
    if k == 'Mississipie':
        print('\nThe Mississipie is slow, but wide.')

for river in rivers_of_the_world.keys():
    print(river)

for country in rivers_of_the_world.values():
    print(country)


polled_people = ['jen', 'prim', 'samuel', 'harry', 'john']

favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python'
    }

for people in polled_people:
    if people in favorite_languages:
        print('Thank you for responding.')
    else:
        print('Please take our poll.')

person = {'first_name': 'Calvin','last_name': 'Howe','age': 12,'city': 'Kansas City'}
person_2 = {'first_name': 'Frodo','last_name': 'Baggins','age': 51,'city':'Hobbiton'}
person_3 = {'first_name': 'Martin','last_name': 'Howe','age': 3,'city': 'Kansas City'}

peoples = [person, person_2, person_3]

for people in peoples:
    print(people)

pet = {'type': 'panther','owner': "Drizzt Do'Urden",'name': 'Guenhaver'}
pet_2 = {'type': 'brown bear','owner': 'Anna','name': 'Wojtek'}
pet_3 = {'type': 'wolf','owner': 'Gunter','name': 'Nacht'}

pets = [pet, pet_2, pet_3]

for pet in pets:
    print(pet)


numbers = {'Calvin': [5, 20, 12, 6, 8, 10],
           'geoffrey': [2, 0],
           'lily': [12],
           'john': [13, 100, 20],
           'brandon': [10]}

print(f"\nCalvin's favorite numbers are {numbers['Calvin']}.")
print(f"\ngeoffrey's favorite numbers are {numbers['geoffrey']}.")
print(f"\nlily's favorite numbers are {numbers['lily']}.")
print(f"\njohn's favorite numbers  are {numbers['john']}.")
print(f"\nbrandon's favorite numbers are {numbers['brandon']}.")

cities = {'city_one':
             {'name': 'Crete',
              'pop': 'unknown',
              'country': 'Greece',
              'fact': 'Crete is the home of the minotaur in Greek myths.'},
         'city_two': {'name': 'Athens',
                      'pop': 'unknown',
                      'country': 'Greece',
                      'fact': 'The Greek hero Theseus ousted Medea here.'},
         'city_three':{'name': 'Thebes',
                       'pop': 'unknown',
                       'country': 'Greece',
                       'fact': 'The city was founded when a cow died, marking the spot.'}}

print(f"\n{cities['city_one']}")
print(f"\n{cities['city_two']}")
print(f"\n{cities['city_three']}")


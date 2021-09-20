pizzas = ['sausage','meat lovers','pepperoni']

for pizza in pizzas:
    print(f"I like {pizza} pizza.")
print('\nI really like pizza.')



big_cats = ['snow leopard','bengal tiger','clouded leopard','cougar','african lion']

for cats in big_cats:
    print(f"A {cats} is one of the biggest cats in the world.")
print("All of these amazing cats could kill you")



numbers = list(range(1,21))
print(numbers)

big_numbers = list(range(1,1_000_001))

for num in big_numbers:
   print(num)


print(min(big_numbers))
print(max(big_numbers))
print(sum(big_numbers))



odd_numbers = list(range(1,20,2))
print(odd_numbers)



sum_three = [value * 3 for value in range(1,11)]
print(sum_three)



cubes = []
for numin in range(1,11):
    cube = numin ** 3
    cubes.append(cube)

print(cubes)

cubess = [numine ** 3 for numine in range(1,11)]
print(cubess)


big_cats = ['snow leopard','bengal tiger','clouded leopard','cougar','african lion']

print('The first three items in the list are:')
for cat in big_cats[:3]:
    print(cat.title())

print('The middle three items in the list are:')
for cat in big_cats[1:4]:
    print(cat.title())

print('The last three items in the list are:')
for cat in big_cats[:-3]:
    print(cat.title())


pizzas = ['sausage','meat lovers','pepperoni']
friends_pizzas = pizzas[:]

pizzas.append('cheese')
friends_pizzas.append('pineapple')

print('My favorite pizzas are:')
for pizza in pizzas:
    print(pizza)

print('My friends favorite pizzas are:')
for pizza in friends_pizzas:
    print(pizza)

for food in pizzas:
    print(food)

for foods in friends_pizzas:
    print(foods)


buffet_foods = ('Beer','Meat','Bread','Cheese','Olives')
#buffet_foods[0] = 'Pie'
print(buffet_foods)

buffet_foods = ('Ale','Beef','Bread','Cheese','Beer')
for buffet in buffet_foods:
    print(buffet)
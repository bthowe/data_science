print('calvin')




calvin = 11
print(calvin)

calvin = calvin * 365
print(calvin)

def boxy(book):
    return book ** 3

print(boxy(4))




tree = [1, 3, 7]
print(tree)
print(tree[2])
print(tree[0])
print(tree[1])

tree[1] = 5
print(tree)

for i in tree:
    print(boxy(i))

x = 1
if x == 1:
    print('math')
else:
    print('redwall')


def greet():
    return 'hello world!'

def main():
    print('Hello World!')


def name(Howe):
    print(Howe)
    return Howe

last_name = name('WThomas')
print(last_name)

def check_alive(health):
    if health > 0:
        return True
    if health <= 0:
        return False

def main(verb, noun):
    return verb + noun

def summation(num):
    sum = 0
    for i in range(num + 1):
        sum = sum + i
    return 'sum + i'

print(summation(10))

sib_list = [11,10,8,7,5,3]

seth_age = sib_list[3]
print(seth_age)


mag_list = sib_list[4]
print(mag_list)

kids = ['calvin','samuel','kay']

oldest = kids[0]
print(oldest)


kids[1] = 'travis'
print(kids)

sib_list[5] = 38
print(sib_list)

print(sib_list[0])
print(sib_list[1])
print(sib_list[5])
print(sib_list[-1])
print(sib_list[-2])
print(sib_list[0:3])
print(sib_list[-3:-1])
print(sib_list[-3:])

for hobbes in sib_list:
    print(hobbes)

for hobbes in sib_list:
    print(hobbes + 5)

sib_list5 = [hobbes + 5 for hobbes in sib_list]
print(sib_list5)

pen = [hobbes + ' howe' for hobbes in kids]
print(kids)
print(pen)

nollij = [hobbes + ' is dumb' for hobbes in kids]
print(nollij)

# 1. make a list called death_ages with the ages each of your siblings will die
# 2. in a for loop, print each of the ages in death_ages plus 1'
# 3. make a list called new_death_ages using a list comprehension, with the ages from death_ages plus 10

death_ages = [100,95,89,91,79,80]

for spiff in death_ages:
    print(spiff + 10)

new_death_ages =[spiff + 10 for spiff in death_ages]
print(new_death_ages)


# 1. make a list called people with three super hero names
# 2. make a function called dumb that adds " is dumb" to the end of a string input
# 3. make a list called new_people using a list comprehension, with the people list and the functino dumb

people = ['superman','metalmouth','flash']

def dumb(name):
    return name + ' is dumb'

new_people = [dumb(name) for name in people]
print(new_people)



# 1. make a list called star_wars with three light saber colors
# 2. make a function called jedi that adds "A Jedi's lightsaber can be " to the beginning of a string input
# 3. make a list called light_saber using a list comprehension, that adds the elements of star_wars to jedi

star_wars = ['green', 'blue', 'purple']

def jedi(weopan):
    return "A Jedi's lightsaber can be " + weopan

light_saber = [jedi(weopan) for weopan in star_wars]
print(light_saber)

light_saber = []
for tracer in star_wars:
    light_saber.append(jedi(tracer))
    # star_wars.append("A Jedi's lightsaber can be " + tracer)
print(light_saber)


howes = ['Kassie', 'Travis', 'Calvin', 'Samuel', 'Kay', 'Seth', 'Martin', 'Maggie']
def family(bullet):
    return bullet + ' is cool'

taggur_ung =[family(bullet) for bullet in howes]
print(taggur_ung)

taggur_ung = []
for suzie in howes:
    taggur_ung.append(family(suzie))
print(taggur_ung)


numbers = range(10)

def interger(cola):
    return cola * 5

new_numbers =[interger(cola) for cola in numbers]
print(new_numbers)

new_numbers = []
for cola in numbers:
    new_numbers.append(interger(cola))
print(new_numbers)

seven = range(8)

for boof in seven:
    if boof == 1 :
        print(1 + 5)
    else:
        print()




def index(array, n):
    if len(array) <= n:
        return -1
    return array[n] ** n

print(index([1, 2, 3, 4], 2))
print(index([1, 2, 3, 4], 4))
print(index([1, 2, 3, 4], 3))


def greet(name):
    return "Hello, "+ name +" how are you doing today?"


print(greet('Calvin'))


def set_alarm(employed, vacation):
    if employed and not vacation: return True
    else: return False

print(set_alarm(True, True))

def arr(n):
    if n:
    # return list(range(n))
        return [ bat for bat in range(n)]
    else:
        return []

print(arr(0))

def array_plus_array(arr1,arr2):
    sum = 0
    for ar in arr1:
        sum = sum + ar
    for er in arr2:
        sum = sum + er
    return sum

print(array_plus_array([1, 2, 3], [4, 5, 6]))



def summer(moss):
    sum = 1
    for de in moss:
        if de == 5:
            sum = sum * de
    return sum
print(summer([5, 4, 6, 8, 2, 5, 5]))



def f(number):
    return number + 5

def g(pig):
    return pig - 5

# create a function called g
# g has to be the inverse of f, such that g(f(x)) = x

print(g(f(15)))
print(g(f(15)) == 15)
print(g(f(55)) == 55)
print(g(f(155)) == 155)




print(f(g(15)))
print(f(g(15)) == 15)
print(f(g(55)) == 55)
print(f(g(155)) == 155)

def f(randy):
    return randy * 8 + 5

def g(matt):
    return (matt - 5) / 8


print(g(f(5)))
print(g(f(15)))

print(f(g(5)))
print(f(g(15)))

gee = g(5)
print(gee)
# print(f(gee))
# gee = g(15)
# print(f(gee))


# create a function called area_rect that returns the area of a rectangle with length and width as arguments
def area_rect(n, r):
    return n * r


import math
# right a function that takes two arguements: one for the number of things and one for the number taken at a time.

print('\n\n\n\n')
print(math.factorial(3))

def torivor(n, r):
    return math.factorial(n) /math.factorial(n - r)
print(torivor(12,4))



# make a list of of numbers 0 through 20

va_lere = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
za_lere = list(range(21))

# create a function that takes a number as an argument, adds five to it and then divides by three

def rand(num):
    return (num + 5) / 3

# make a for loop that iterates over your list and prints the output from your function with the forloop

for n in va_lere:
    print(rand(n))

print([rand(n) for n in va_lere])

def rogue(x):
    return 3 ** -x

print(rogue(0))


f = lambda x: 3 ** -x

import numpy as np
import matplotlib.pyplot as plt

def war_lock(jimmy):
    x = np.linspace(-2, 2, 1000)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, jimmy(x))
    plt.show()


# war_lock(lambda x: (1 / 2) ** (2 * x) )
# war_lock(lambda x: (1 / 2) ** (-x -2) )

pally = lambda y: y ** (1 / 2)
print(pally(9))

# war_lock( lambda y: y ** (1 / 2))

def war_lock2(jimmy, x):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, jimmy(x))
    plt.show()
# war_lock2(lambda y: y ** (1 / 2), np.linspace(0, 49, 1000))

def war_lock3(jimmy, x_min, x_max):
    x = np.linspace(x_min, x_max, 1000)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, jimmy(x))
    plt.show()
# war_lock3(lambda y: y ** (1 / 2), x_min=0, x_max=64)





def plotter(f, x_min, x_max):
    """
    This function plots a function, f, over the x interval [x_min, x_max]

    :param f: a function
    :param x_min: int
    :param x_max: int
    """
    x = np.linspace(x_min, x_max, 1000)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, f(x))
    plt.show()


import numpy as np
# plotter(np.cos, -10, 10)
# plotter(np.sin, -10, 10)
# plotter(np.tan, -10, 10)

print(np.sin(0))
print(np.cos(0))
print(np.sin(np.pi))
print(np.cos(np.pi))
print(np.sin(np.pi / 2))
print(np.cos(np.pi / 2))

print(np.arcsin(1) == np.pi / 2)




a = list(range(100))
# iterate through a using a forloop
# within the for loop,
#   1. if even, then print "Even"
#   2. if odd, then print "Odd"
#   3. if even and 10, print "Puffy"
#   4. if even or 11, print "Poofy"

for r in a:
    print(r)
    if r % 2 == 0:
        print('Even')
    if r % 2 == 1:
        print('Odd')
    if r % 2 == 0 and r == 10:
        print('Puffy')
    if r % 2 == 0 or r == 11:
        print('Poofy')
    print('\n')




fam = ['Kassie', 'Travis', 'Calvin', 'Samuel', 'Kay', 'Seth', 'Maggie', 'Martin']
# iterate through fam using a for loop
# if the person is a boy print "boy", otherwise print "girl"

for jam in fam:
    if jam in ['Travis','Calvin','Samuel','Seth','Martin']:
        print(' boy')
    else:
        print(' girl')


fam2 = ['Kassie', 'Travis', 'Calvin', 'Samuel', 'Kay', 'Seth', 'Maggie', 'Martin', 'Martian']
# iterate through fam2 using a for loop
# if the person is a boy print "boy", if a girl print "girl", if a martian print "Martian"

for bam in fam2:
    if bam in ['Travis','Calvin','Samuel','Seth','Martin']:
        print(' boy')
    elif bam == 'Martian':
        print(' Martian')
    else:
        print(' girl')







# make a function that calculates the log of x given a base
# log_b(x) = y => find y such that b ** y = x

def litres(time):
    return int(0.5 * time)
print(litres(11.8))

print(litres)





def get_average(marks):
    sum = 0
    for i in marks:
        sum = sum + i
    return sum / len(marks)

def get_average2(marks):
    sum = 0
    count = 0
    for i in marks:
        sum = sum + i
        count = count + 1
    return sum / len(marks)


print(get_average([2, 3, 4, 5, 6]))


a_list = range(10)

# create a for loop and count the number of iterations through the list a_list

print('\n\n')
s = 0
for count in a_list:
    s += 1
    # s = s + 1
print('sum:', s)



# using a_list, iterate using a for loop over its elements
# define a variable that will be the total
# if even, add the number
# if odd, subtract the number
# print the total
z = 0
for t in a_list:
    if t % 2 == 0:
        z = t + z
    else:
        z = z - t
print(z)


for i, val in enumerate(a_list):
    print(i, val)


# in each iteration of the while loop, print the value of a and then add one to it
a = 0
while a < 10:
    print(a)
    a += 1
    print(a)
    print('\n')

for num in range(10):
    print(num + 1)

# a = 0
# while a < 10:
#     if a != 5:
#         print(a)
#         a += 1
#     print(a)


# ==
# <
# >
# !=


# make a for loop that loops over b = range(15)
# count the number of iterations
# print the count if the iteration number is less than five
# print -1 if the iteration number is between six and ten
# print 0 if the iteration number is greater than ten

b = range(15)

k = 0
for count in b:
    k += 1
    if k <= 5:
        print('sum:', k)
    elif 10 >= k >= 6:
        print(-1)
    else:
        print(0)

# if (10 >= k) and (k >= 6)


# create a function called check_exam that takes two inputs: the first is a list of correct answers, the second is a list of guesses
# count the number "correct"
print('\n')
def check_exam(actual, guesses):
    for counte in range(len(actual)):
        print(counte)


check_exam(['a', 'b', 'c'], ['a', 'a', 'c'])
# print(check_exam(['a', 'b', 'c'], ['a', 'a', 'c']))
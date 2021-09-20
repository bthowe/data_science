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
         print(f"\n{self.first_name} "
               f"{self.last_name}"
               f" is {self.age}"
               f" years old, {self.height}"
               f" tall, and {self.richness} rich")

def greet_user(self):
        print(f"Hello {self.first_name} {self.last_name}.")

def increment_login_attempts(self, login):
        self.login_attempts += login
        print(f"You have tried logging in {login}.")



print("Give me two numbers and I will multiply them")
print("Enter q to quit")

while True:
     first_number = input("First number: ")
     if first_number == 'q':         break
     second_number = input("Second number: ")
     if second_number == 'q':
         break
     try:
         answer = float(first_number) * float(second_number)
     except ValueError:
         print('Give the numbers in numerical form. You cannot use words')
     else:
         print(answer)
filename = ('cats.txt', 'dogs.txt')
for file in filename:
     try:
         with open(file) as file_object:
             lines = file_object.readlines()
     except FileNotFoundError:
         print(f"The file {filename} does not exist.")
     else:
         print(lines)



with open('moby_dick.txt') as moby_dick:
    # list_of_lines = moby_dick.readlines()
    list_of_lines = moby_dick.read()

print(list_of_lines.count('the '))



import json


favorite_number = input("Tell me your favorite number. ")


filename = 'favorite_numbers.json'
with open(filename, 'w') as f:
    json.dump(favorite_number, f)

import json

filename = 'favorite_numbers.json'
with open(filename) as f:
    favorite_number = json.load(f)


print(f"Your favorite number is {favorite_number}!")
print()





import json

filename = 'favorite_number.json'

try:
    with open(filename) as f:
        favorite_number = json.load(f)
except FileNotFoundError:
    favorite_number = input("Tell me your favorite number. ")
    with open(filename, 'w') as f:
        json.dump(favorite_number, f)
        print(f"Is this your favorite number? {favorite_number}.")
        print(f"Your favorite number is {favorite_number}!")
else:
    print(f"I already knew that your favorite number is {favorite_number}!")






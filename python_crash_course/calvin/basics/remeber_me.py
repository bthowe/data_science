import json

def get_stored_username():
    filename = 'username.json'
    try:
        with open(filename) as f:
            username = json.load(f)
    except FileNotFoundError:
        return None
    else:
        return username


def get_new_username():
    username = input('What is your name? ')
    filename = 'username.json'
    with open(filename, 'w') as f:
        json.dump(username, f)
    return username


def greet_user():
    username = get_stored_username()
    if username:
        username_true = input(f'Is {username} the correct username? ')
        if username_true == 'y':
            print(f'Welcome back, {username}!')
        else:
            username = get_new_username()
            print(f"We'll remember you when you come back, {username}!")
    else:
        get_new_username()


def greet_user2():
    filename = 'username.json'
    try:
        with open(filename) as f:
            username = json.load(f)
    except FileNotFoundError:
        username = input("What is your name? ")

    if username == 'username.json':
        print(f"Welcome back, {username}.")
    else:
        print(f"Welcome back, {username}.")

        with open(filename, 'w') as f:
            json.dump(username, f)
            print(f"We'll remember you when you come back, {username}!")


greet_user()
#
#
# def display_message():
#     print("I am learning about functions!")
# display_message()
#
#
# def favorite_book(book):
#     print(f"\n{book} is one of my favorite books.")
# favorite_book('The Hunger Games')
#
#
# def make_shirt(size='small', message="Storm King's Thunder"):
#     print(f"The size of shirt is {size}, and it should say '{message}'.")
# make_shirt('small', 'D&D')
# make_shirt('small','I Make History')
#
#
# def describe_city(city, country='England'):
#     print(f"{city} is in {country}.")
#
# describe_city(city='London')
# describe_city(city='York')
# describe_city(city='Dublin',country='Ireland')
#
#
# def city_country(city,country):
#     print(f"{city.title()}, {country.title()}")
#
# city_country(city='athens',country='greece')
# city_country(city='berlin',country='germany')
# city_country(city='crete',country='greece')
#
#
# def make_album(name,album_name):
#     album = {'author': name,'album': album_name}
#     return album
# albume = make_album('Piano Guys','Limitless')
# print(albume)
#
# while True:
#     print("\nPlease tell me who wrote the album and what it's name is.")
#     p_name = input("Your name. ")
#     if p_name == 'q':
#         break
#     a_name = input("Album name. ")
#
#     names = make_album(p_name,a_name)
#     print(f"You made an album by {p_name} called {a_name}")
#
#
# texts = ['Shut your mouth, your breath stinks.',"I'm almost done.",'Just so you know, I drew a mustache on you in pen.']
#
# def show_messages(text):
#
#     for texte in text:
#         mfm = texte
#         print(mfm)
#
# show_messages(texts)
#
#
# texts = ['Shut your mouth, your breath stinks.',"I'm almost done.",'Just so you know, I drew a mustache on you in pen.']
# sent_texts = []
#
# def send_messages(text):
#
#     for texte in text:
#         mfm = texte
#         print(mfm)
#
# show_messages(texts)
#
# while texts:
#     sending_texts = texts.pop()
#     sent_texts.append(sending_texts)
#
# print("\nThe following texts have been printed.")
# for sent_text in sent_texts:
#     print(sent_text)
#
#
# send_messages(text=['Shut your mouth, your breath stinks.',"I'm almost done.",'Just so you know, I drew a mustache on you in pen.'])
#
#
# def make_sandwich(*ingredients):
#     print("\nMaking your sandwich, with your following ingredients:")
#     for ingredient in ingredients:
#         print(f" - {ingredient}")
#
# make_sandwich('bread','honey','peanut butter','baloney','turkey','ham','vegamite')
# make_sandwich('honey','a little bit of bread','a little bit of peanut butter')
# make_sandwich('bread','nutella')
#
#
# def build_profile(first, last, **user_info):
#     user_info['first_name'] = first
#     user_info['last_name'] = last
#     return user_info
#
# user_profile = build_profile('calvin', 'howe',
#                              location='north america',
#                              favorite_book='Spearhead',
#                              field='math')
# print(user_profile)
#
#
# def car_making(model_name, manufacturer, **notes):
#     notes['model'] = model_name
#     notes['maker'] = manufacturer
#     return notes
#
# car = car_making('ferrari','ferrari',
#                  color='cherry red',
#                  tow_package='false')
# print(car)


import unittest

def country_city(city, country, population=''):
    if population:
        place = f"{city}, {country}, population {population}."
        print(place)
    else:
        print(f"{city}, {country}.")

country_city('Athens', 'Greece', population=7000000)

if __name__ == '__main__':
    unittest.main()



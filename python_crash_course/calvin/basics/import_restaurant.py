from restaurants import Restaurants

my_restaurants = Restaurants('Red Lobster', 'lobster', 76)
print(my_restaurants.food())

my_restaurants.food()
my_restaurants.set_number_served(people=9999999)
my_restaurants.place_name()
my_restaurants.increment_number_served(0)


from admin_users_privileges import Users, Privileges, Admin

admins = Admin('Jeff', 'Feeshel', 40, "6'7", '$7900000000','stuff')

admins.privileges.show_privileges()

from user import Users
from admin_users_privileges import Privileges, Admin

adymin = Admin('Jeff', 'Feeshel', 49, "6'7", '$8000000000', 'stuff')

adymin.privileges.show_privileges()


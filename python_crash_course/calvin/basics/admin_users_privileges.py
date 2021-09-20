from user import Users

class Privileges:

    def __init__(self, privileges):
        self.privileges = privileges

    def show_privileges(self):
        self.privileges = ['can delete posts', 'can add posts', 'can ban users', 'can appoint deputy admins']
        print(f"I, Admin, {self.privileges}")

class Admin(Users):

    def __init__(self, first_name, last_name, age, height, richness, privilege):
        super().__init__(first_name, last_name, age, height, richness)
        self.privileges = Privileges(privilege)

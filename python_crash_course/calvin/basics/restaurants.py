class Restaurants:
    def __init__(self, name, cuisine, number_served):
        self.name = name
        self.cuisine = cuisine
        self.number_served = number_served
        self.number_served = 0
    """Print the name of this restaurant"""
    def place_name(self):
        print(f"This restaurant is named {self.name}.")

    """Print what kind of food this restaurant has"""
    def food(self):
        print(f"This restaurant has {self.cuisine} food.")

    """Print how many customers have been served"""
    def set_number_served(self, people):
        self.number_served = people
        print(f"This restaurant has served {self.number_served} people.")

    def increment_number_served(self, number):
        self.number = number
        self.number_served += number
        print(f"{self.number}")





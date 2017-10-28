class Bag:

    def __init__(self):
        self.gift_name_list = list([])
        self.gift_weight_list = list([])
        self.total_weight = 0.0

    def add_gift_item(self,gift_name,gift_weight):
        self.gift_name_list.append(gift_name)
        self.gift_weight_list.append(gift_weight)

    def remove_gift_item(self,gift_name,gift_weight):
        self.gift_name_list.remove(gift_name)
        self.gift_weight_list.remove(gift_weight)

    def calculate_total_weight(self):
        self.total_weight = sum(self.gift_weight_list)


class RandomSolution:

    def __init__(self):
        self.bag_list = list([])
        self.total_solution_value = 0.0

    def add_bag(self,bag):
        self.bag_list.append(bag)

    def add_bag_list(self,bag_list):
        self.bag_list.extend(bag_list)

    def remove_bag(self,bag):
        self.bag_list.remove(bag)

    def remove_bag_list(self,bag_list):
        for bag in bag_list:
            self.bag_list.remove(bag)

    def calculate_solution_value(self):
        self.total_solution_value = 0.0
        for bag in self.bag_list:
            self.total_solution_value += bag.total_weight


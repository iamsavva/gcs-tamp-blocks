from colorama import Fore
import typing as T

def ERROR(*texts):
    print(Fore.RED + " ".join([str(text) for text in texts]))

def WARN(*texts):
    print(Fore.YELLOW + " ".join([str(text) for text in texts]))

def INFO(*texts):
    print(Fore.BLUE + " ".join([str(text) for text in texts]))

def YAY(*texts):
    print(Fore.GREEN + " ".join([str(text) for text in texts]))

def all_possible_combinations_of_items(item_set: T.List[str], num_items: int):
    """
    Recursively generate a set of all possible ordered strings of items of length num_items.
    """
    if num_items == 0:
        return [""]
    result = []
    possible_n_1 = all_possible_combinations_of_items(item_set, num_items-1)
    for item in item_set:
        result += [ item + x for x in possible_n_1 ]
    return result
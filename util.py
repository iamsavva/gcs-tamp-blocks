from colorama import Fore

def ERROR(*texts):
    print(Fore.RED + " ".join([str(text) for text in texts]))


def WARN(*texts):
    print(Fore.YELLOW + " ".join([str(text) for text in texts]))


def INFO(*texts):
    print(Fore.BLUE + " ".join([str(text) for text in texts]))


def YAY(*texts):
    print(Fore.GREEN + " ".join([str(text) for text in texts]))
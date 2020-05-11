VERBOSITY = 10

NO = 0
GENERAL = 10
SPECIFIC = 20

def print_specific(msg):
    if VERBOSITY == SPECIFIC:
        print(msg)

def print_general(msg):
    if VERBOSITY == GENERAL:
        print(msg)
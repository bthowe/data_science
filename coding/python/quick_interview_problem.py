import sys
from collections import Counter

def has_chars(s1, s2):
    boolean = True

    counter_s1 = Counter(s1)
    counter_s2 = Counter(s2)
    for letter2 in s2:
        if not counter_s2[letter2] <= counter_s1[letter2]:
            boolean = False

    return boolean


def has_chars3(s1, s2):
    return all([False if not Counter(s2)[letter2] <= Counter(s1)[letter2] else True for letter2 in s2])


def main():
    s1 = "waterfall"
    s2 = "taller"
    print(has_chars3(s1, s2))  # True

    # Example 2
    s1 = "waterfall"
    s2 = "feel"
    print(has_chars3(s1, s2))  # False


if __name__ == '__main__':
    main()

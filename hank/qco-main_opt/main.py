import sys

import scripts




def read_options(args):
    args = args[1:]
    options = {}
    is_argument = False

    for i in range(len(args)):
        if is_argument:
            is_argument = False
            continue

        key = args[i][1:]
        value = True

        if i + 1 < len(args) and args[i + 1][0] != '-':
            value = args[i + 1]
            is_argument = True

        options[key] = value

    return options


if __name__ == "__main__":
    options = read_options(sys.argv)
    scripts.sample_norms(**options)

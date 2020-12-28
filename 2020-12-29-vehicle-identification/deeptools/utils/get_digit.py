import re


def get_digit(fileName):
    regex = re.compile('\d+')
    list = regex.findall(fileName)

    if len(list) > 0:
        return list[0]
    else:
        return ""

def AreThereDuplicateValues(listi):
    return len(listi) != len(set(listi))

def FindElementInList(element, list):
    for i,val in enumerate(list):
        if element == val:
            return i
    return None

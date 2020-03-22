def create_dict(*idxes):
    '''
        create nested dictionary with the given idxes
    '''

    height  = len(idxes)
    output = {}

    stack  = []
    stack.append(output)

    for depth in range(height):
        stack_temp = []
        while len(stack) > 0:
            cur_elmt = stack.pop()
            for idx in idxes[depth]:
                cur_elmt[idx] = {}
                stack_temp.append(cur_elmt[idx])
        stack = stack_temp

    return output

        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError("Can't convert 'str' object to 'boolean'")



def describe_dict(input_dict):
    for k,v in input_dict.items():
        if type(v)==dict:
            print(k, len(v))
        else:
            print(k, v.shape)
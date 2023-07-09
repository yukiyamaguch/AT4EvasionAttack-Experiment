from yaml import safe_load
import numpy as np

def calc_AP(t):
    if t["NODE"] == "Root":
        return sum([t["WEIGHTS"][i]*calc_AP(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "AEA":
        return sum([t["WEIGHTS"][i]*calc_AP(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "Scenario":
        return calc_AP(t["CHILDREN"][0]) * calc_AP(t["CHILDREN"][1])
    elif t["NODE"] == "AEML":
        return max([calc_AP(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "AEM":
        return t["ERR"]*t["AP"]
    elif t["NODE"] == "CAL":
        return np.prod([calc_AP(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "CA":
        return t["AP"]
    else:
        print("error")
        print(t)
        exit()

with open('at4ea.yaml', 'r') as yml:
    tree = safe_load(yml)

print(f'Root:     {calc_AP(tree)}')
print(f'Digital:  {calc_AP(tree["CHILDREN"][0])}')
print(f'Physical: {calc_AP(tree["CHILDREN"][1])}')

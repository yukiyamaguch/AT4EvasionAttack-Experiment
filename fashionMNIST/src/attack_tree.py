from yaml import safe_load
import numpy as np

def calc_AR(t):
    if t["NODE"] == "Root":
        return sum([t["WEIGHTS"][i]*calc_AR(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "Attribute":
        return sum([t["WEIGHTS"][i]*calc_AR(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "Scenario":
        return calc_AR(t["CHILDREN"][0]) * calc_AR(t["CHILDREN"][1])
    elif t["NODE"] == "AIAttackList":
        return t["AR"] * max([calc_AR(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "AIAttack":
        return t["ASR"]
    elif t["NODE"] == "ConvAttackList":
        return np.prod([calc_AR(t["CHILDREN"][i]) for i in range(len(t["CHILDREN"]))])
    elif t["NODE"] == "ConvAttack":
        return t["AR"]
    else:
        print("error")
        print(t)
        exit()

with open('attack_tree.yaml', 'r') as yml:
    tree = safe_load(yml)

print(calc_AR(tree))

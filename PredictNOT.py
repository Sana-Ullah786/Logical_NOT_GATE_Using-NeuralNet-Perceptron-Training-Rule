import numpy as np,pickle
from prettytable import PrettyTable

def activationFunction(linearUnit):
    if linearUnit <= 0:
        return 0
    else:
        return 1


def notGate(Input, weights, bias):
    return activationFunction(np.dot(weights, Input)+bias)


def printTable(a,output):
    myTbale=PrettyTable(["a","NOT"])
    myTbale.add_row([a,output])
    print(myTbale)
if __name__ == '__main__':
    with open('NOT.pkl','rb') as f:
            parameters=pickle.load(f)
    while True:
        try: 
            a=input("Enter a (N to Exit) = ")
            if(int(a)<0 or int(a)>1  ):
                print("Invalid Inputs")
                continue
            printTable(a, notGate(np.array([int(a)]),parameters[0],parameters[1]))
        except:
            break
    
    


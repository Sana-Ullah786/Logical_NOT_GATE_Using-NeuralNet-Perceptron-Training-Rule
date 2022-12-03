
import numpy as np,pickle


def activationFunction(linearUnit):
    if linearUnit <= 0:
        return 0
    else:
        return 1


def notGate(Input, weights, bias):
    return activationFunction(np.dot(weights, Input)+bias)


def updateBias(lr, loss):
    return lr*loss


def updateWeights(lr, loss, input):
    return lr*loss*input


def saveModel(w,b):
    parameters=[w,b]
    with open('NOT.pkl','wb') as f:
        pickle.dump(parameters,f)


def trainModel(X, y, w, b, lr):
    flag = True
    while flag:
        for i in range(len(X)):
            output = notGate(X[i], w, b)
            if(output != y[i]):
                w += updateWeights(lr, (y[i]-output), X[i])
                b += updateBias(lr, (y[i]-output))
                break
        else:
            break
    print("\nModel Trained Successfully and Saved")
    saveModel(w,b)
    
    


if __name__ == '__main__':
    X = np.array([[1],[0]])
    y = [0, 1]
    w = np.array([0.0])
    lr = 0.1
    b = -0.7
    trainModel(X, y, w, b, lr)
    

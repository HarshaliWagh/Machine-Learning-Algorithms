import sys
import random
import math

def gra_descent(data,trainLabels,rows,cols,eta):

    w = []
    for i in range(0, cols):
        w.append(0.02 * random.random() - 0.01)

    emp= 0
    diff = 1.0

    while (diff > Stop_Condition):
        dellf = [0] * cols

        for i in range(0, rows, 1):
            if (trainLabels.get(i) != None):
                a = trainLabels[i] * dot(w, data[i])
                for j in range(0, cols):
                    if a<1:
                        dellf[j] += -(trainLabels[i] * data[i][j])
                    else:
                        dellf[j] += 0

        for j in range(0, cols, 1):
            w[j] -= eta * dellf[j]

        prev = emp
        emp= 0

        for i in range(0, rows):
            if (trainLabels.get(i) != None):
                emp += max(0, 1 - (trainLabels.get(i)) * dot(w, data[i]))
            diff = abs(prev - emp)

        #print('Objective :', str(emp_risk))
    return w

def dot(a, b):

    list1 = []
    for x, y in zip(a, b):
        mul = x * y
        list1.append(mul)
    return sum(list1)

if __name__ == '__main__':

    datafile = sys.argv[1]

    f = open(datafile)
    data = []
    i = 0
    l = f.readline()

    while (l != ''):
        a = l.split()
        l2 = []
        for j in range(0, len(a), 1):
            l2.append(float(a[j]))
        data.append(l2)
        l = f.readline()

    labelfile = sys.argv[2]
    f = open(labelfile)

    trainLabels = {}
    l = f.readline()

    while (l != ''):
        a = l.split()
        trainLabels[int(a[1])] = int(a[0])
        l = f.readline()

    for i in trainLabels:
        if trainLabels[i] == 0:
            trainLabels[i] = -1

    for i in range(0, len(data)):
        data[i].append(1)

    rows = len(data)
    cols = len(data[0])
    eta = float(sys.argv[3])
    Stop_Condition = float(sys.argv[4])
    #eta = 0.001

    w = gra_descent(data, trainLabels, rows, cols, eta)

    #print('W :',w)
    norm= 0
    for i in range(0, cols - 1):
        norm += w[i] ** 2
    norm = math.sqrt(norm)
    dist = abs(w[len(w) - 1] / norm)
    #print('Origin distance:' + str(dist))

    for i in range(0, rows):
        if (trainLabels.get(i) == None):
            dp = dot(w, data[i])
            if (dp > 0):
                print('1', i)
            else:
                print('0', i)

import sys
import math
import random

#Read Data from file
filename = sys.argv[1]
f = open(filename)
data = []
i=0
l = f.readline()
while(l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    l2.append(1)
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()

#Read labels from file
filename = sys.argv[2]
f = open(filename)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()

while(l != ''):
    a = l.split()
    if(int(a[0]) == 0):
        trainlabels[int(a[1])] = -1
    else:
        trainlabels[int(a[1])] = int(a[0])
    l = f.readline()
    n[int(a[0])] += 1

#Initialize w and dellf
w = []
dellf = []
for j in range(0,cols,1):
    w.append(random.uniform(-0.01, 0.01))
    dellf.append(0)
#print ("Intial W:",w)

def dot(x, y):
    return sum(x_i*y_i for x_i, y_i in zip(x, y))

#Gradient descent iteration

eta = 0.0001
stopping_condition= 0.001
preverror = float ('inf')
error = 0

for i in range(0, rows, 1):
    if(trainlabels.get(i)!= None):
        error += ((trainlabels[i] - dot(w,data[i]))**2)
#print (error)

while (preverror - error > stopping_condition):
    preverror = error
    
    for j in range(0, cols, 1):
        dellf[j] = 0
    
    for i in range(0, rows, 1):
        if(trainlabels.get(i) != None):
            dp = dot(w,data[i])
            for j in range(0, cols, 1):
                dellf[j] += (trainlabels[i] - dp)*(data[i][j])

    for j in range(0, cols, 1):
        w[j]=w[j]+eta*(dellf[j])

    error = 0
    for i in range(0, rows, 1):
        if(trainlabels.get(i)!= None):
            error += ((trainlabels[i] - dot(w,data[i]))**2)
#print ("Final W",w)
normw = 0

for j in range(0, cols-1, 1):
    normw += w[j]**2
normw = math.sqrt(normw)
#print ("||w||=", normw)

dist_origin = abs(w[len(w)-1]/normw)
#print ("distance to orgin=",(dist_origin))

#Prediction
for i in range(0,rows,1):
    if(trainlabels.get(i)==None):
        dp=dot(w,data[i])
        if(dp>0):
            print("1", i)
        else:
            print("0", i)

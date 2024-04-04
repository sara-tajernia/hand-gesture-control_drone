x= []
for i in range(10):
    y=[]
    for j in range(10):
        z= []
        for k in range(3):
            z.append(k+1)
        y.append(z)
    x.append(y)

print(x)

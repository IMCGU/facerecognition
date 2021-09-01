# def fun(a, b=20, *, kw1, kw2=40):
    # #print(a, b, kw1, kw2)
    # print(b,,kw1)

# #fun(1, 2, kw1=3, kw2=4)  # 1 2 3 4
# fun(10, kw1=30)  # 10 20 30 40　　
x=10
outer(x)
def outer(x):
    y=20
    inner(x)
    def inner(x):
        z=30    
        print(x)  
   　　 print('y',y)  
   
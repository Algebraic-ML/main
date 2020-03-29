#一个和图无关的python试验板
import ML_graph as mg
import numpy as np


G = mg.ML_Graph()
t1 = mg.Term(np.matrix([[1, -1], [1, -1]]), True, 't1')
c1 = mg.Constant((0, 0), 1, 'c1')

print(t1 <= c1)

G.addNode(t1).addNode(c1)
print(G.ConstantList)
G.removeNode(c1)
print(G.ConstantList)



# class test1:
#     def __init__(self, data):
#         self.data = data

#     def add1(self):
#         self.data += 1 
#         return self

# class test2(test1):
#     def __init__(self, data):
#         super().__init__(data) 

#     def add1(self):
#         super().add1()
#         self.data += 1
#         return self

# print(test2(0).add1().data)



# t1 = Graph.Term(np.matrix([[1, -1], [1, -1]]), 't1')
# c1 = Graph.Constant((0, 0), 1, 'c1')

# print(t1 <= c1)



# a = np.eye(4)
# print(a[(1,1)])


# G_ML = Graph.ML_Graph()
# print(id(G_ML.Dual()))
# print(id(G_ML.Dual()))
# print(id(G_ML.Dual()))

'''
a = 1
b = 2
c = 3

x = [a,b,c]
y = [b,c]
for t in y:
    if t not in x:
        x.append(t)     
print(x)


class base:
    def __init__(self):
        self.data = None
        
    def __lt__(self, other):
        return self.data > other.data



class NewClass(base):
    def __init__(self, data=None):
        super().__init__
        self.data = data

a = NewClass(3)
b = NewClass(4)
print(not(a < b) )

import random
b = [1,2,3,4,2]
a = random.choice(b)
print(a)
'''
# list1 = ["a","d","c"]

# list2 = ["a","c","d","c","a"]

# print(set(list1).issubset(set(list2)))





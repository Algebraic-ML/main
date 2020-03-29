from Graph import *

import numpy as np
import ML_graph as mg
import matplotlib.pyplot as plt

Graph = mg.ML_Graph()
T1 = mg.Term(np.array([[0, 1], [0, 1]]), True, 'T1+')
T2 = mg.Term(np.array([[1, 0], [1, 0]]), True, 'T2+')
T3 = mg.Term(np.array([[0,1], [1,0]]), False, 'T1-')
T4 = mg.Term(np.array([[1,0], [1,1]]), False, 'T2-')
T5 = mg.Term(np.array([[1,1], [1,0]]), False, 'T3-')
Graph.addNode(T1).addNode(T2).addNode(T3).addNode(T4).addNode(T5)
Graph.initialize()


print(Graph.getAllNodeID())
print(Graph.dualgraph.getAllNodeID())
print(Graph.dualgraph.getNodesByID('T1+_dual'))
print([x.ID for x in Graph.Tr(T1)])

# Graph.paint(figsize=(18,10),node_size=5000,width=2,font_size=12)
# Graph.paint2(figsize=(18,10),node_size=5000,width=2,pic_size=0.07)

Graph.EnforceNegativeTrace()
v = Graph.getNodesByID('v')
Graph.EnforcePositiveTrace()
Graph.paint2(figsize=(18,10),node_size=5000,width=2,pic_size=0.07, figure_name="original figure")
 
i = 0
for term in Graph.PositiveTerm:
    i += 1
    print("Now in term ", i, ": ", term)
    Graph.SparseCrossing(term,v)
    Graph.paint2(figsize=(18,10),node_size=5000,width=2,pic_size=0.07,figure_name="term "+str(i))

plt.show()

# Graph.AtomSetReduction()
# Graph.paint(figsize=(18,10),node_size=5000,width=2,font_size=12)
# Graph.AtomSetReductionDual()

# Rp = Graph.GenerationOfPinning()

#Graph.paint(figsize=(18,10),node_size=5000,width=2,font_size=12)
#Graph.AtomSetReduction()
#
#Graph.paint(figsize=(18,10),node_size=5000,width=2,font_size=12)

#
#Graph.AtomSetReduction()
#
#Graph.paint(figsize=(18,10),node_size=5000,width=2,font_size=12)

#Graph.EnforceNegativeTrace()
"""
Heuristic:
1. preset number of quiries
2. preset limitation of cost
3. preset percent of unlabel pool is labeled
4. preset running time (CPU time) is reached
Formal:
5. the accuracy of a learner has reached a plateau
6. the cost of acquiring new training data is greater than the cost of the errors made by the current model
"""
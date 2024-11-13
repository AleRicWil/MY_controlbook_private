import numpy as np

A = np.array([[1,2],
              [3,4]])

Cr = np.array([[1, 0]])

A1 = np.vstack((np.hstack((A, np.zeros((len(A), 1)))),
                np.hstack((Cr, np.zeros((len(Cr), 1))))))

print(A1)
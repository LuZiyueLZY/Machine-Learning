import numpy as np

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0200058W(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
    """

    InvXTX = np.linalg.inv(X.T @ X)
    b = X.T@y
    w = InvXTX@b

    # return in this order
    return InvXTX, w

X = np.array([[3,2],[-1,4],[5,6]])
y = np.array([1,2,3])

print(A1_A0200058W(X, y))
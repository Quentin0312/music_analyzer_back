import numpy as np


def softmax(z):
    assert len(z.shape) == 2

    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


x1 = np.array([[1, 7, 3, 6]])
print(softmax(x1))
print(softmax(x1)[0].tolist())
test_list = softmax(x1)[0].tolist()
test_list.sort()
print(test_list[-1])
print(softmax(x1)[0].tolist().index(test_list[-1]))

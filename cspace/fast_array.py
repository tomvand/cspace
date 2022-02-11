from array import array

# https://stackoverflow.com/questions/9490628/fast-2-dimensional-array-matrix-in-python-without-c-extensions


class FastArray(object):
    def __init__(self, shape):
        self.shape = shape
        self.data = array("b", [0] * shape[0] * shape[1])
        self.row_size = shape[1]

    def __getitem__(self, index):
        return self.data[index[0] * self.row_size + index[1]]

    def __setitem__(self, index, value):
        self.data[index[0] * self.row_size + index[1]] = value

    def copy(self):
        a = FastArray(self.shape)
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                a[r, c] = self[r, c]
        return a

import numpy as np

def save_vector(output_path, v):
    """
    Save a vector or matrix. This is mainly used to save individual ODFs or response functions.
    These can then be displayed in MRtrix using the `shview` command.
    :param output_path: Path to write text file to.
    :param v: Vector or matrix to write.
    :return: None
    """
    assert len(v.shape) <= 2
    with open(output_path, 'w') as f:
        if len(v.shape) > 1:
            for row in v:
                for elem in row:
                    f.write(str(elem) + ' ')
                f.write('\n')
        else:
            for elem in v:
                f.write(str(elem) + ' ')
        f.close()

def load_vector(input_path):
    """
    Read a vector or matrix. This allows reading response functions and ODFs from MRtrix.
    :param input_path: Input file path name.
    :return: Vector or matrix in stored in the file.
    """
    res = []
    with open(input_path) as f:
        line = f.readline()
        while line:
            if '#' not in line:
                line = line.strip('\n')
                line = line.split(' ')
                line = list(filter(lambda e: e != '', line))
                line = list(map(lambda s: float(s), line))
                res.append(line)
            line = f.readline()
    return np.array(res, dtype=np.float32)
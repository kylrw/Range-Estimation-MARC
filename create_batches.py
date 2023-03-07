from scipy.io import loadmat
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle

def get_chunks(x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the input x and y arrays on the first axis if the y value increases by more than 0.1
    """
    x = x.T[:, :3]
    y = y.T
    chunks = []
    chunk_x = []
    chunk_y = []
    for i in range(x.shape[0]):
        if i == 0:
            chunk_x.append(x[i])
            chunk_y.append(y[i])
        else:
            if i > 0 and y[i] - y[i - 1] > 0.1:
                chunks.append((np.array(chunk_x), np.array(chunk_y)))
                chunk_x = []
                chunk_y = []
            chunk_x.append(x[i])
            chunk_y.append(y[i])
    chunks.append((np.array(chunk_x), np.array(chunk_y)))
    return chunks

def main() -> None:
    #grabs all .mat files from Data/New MATLAB Data
    mat_files = list(Path("Data/New MATLAB Data").glob("*.mat"))

    chunks = []
    for mat_file in mat_files:
        mat_data = loadmat(mat_file)
        chunks.extend(get_chunks(mat_data["X"], mat_data["Y"]))

    #print the number of chunks
    print(len(chunks))

    pickle.dump(chunks, open("data.pkl", "wb"))

if __name__ == '__main__':
    main()
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt

def get_chunks(x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the input x and y arrays on the first axis if the y value increases by more than 0.75
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
        elif i > 0 and y[i] - y[i - 1] > 0.5 and y[i] > 0.8:
            chunks.append((np.array(chunk_x), np.array(chunk_y)))
            chunk_x = []
            chunk_y = []
        else:
            chunk_x.append(x[i])
            chunk_y.append(y[i])

    
    chunks.append((np.array(chunk_x), np.array(chunk_y)))
    return chunks

def main() -> None:
    #grabs all .mat files from Data/New MATLAB Data
    mat_files = list(Path("Data/1").glob("*.mat"))

    chunks = []
    for mat_file in mat_files:
        mat_data = loadmat(mat_file)
        chunks.extend(get_chunks(mat_data["X"], mat_data["Y"]))


    # joins chunks thats initial value is < 0.5 and their previous chunk
    for i, chunk in enumerate(chunks):
        if chunk[1][0] < 0.5 and i > 0:
            chunks[i - 1] = (np.concatenate((chunks[i - 1][0], chunk[0])), np.concatenate((chunks[i - 1][1], chunk[1])))
            del chunks[i]

    # joins chunks thats final value is > 0.5 and their following chunk
    for i, chunk in enumerate(chunks):
        if chunk[1][-1] > 0.5 and i < len(chunks) - 1:
            chunks[i + 1] = (np.concatenate((chunk[0], chunks[i + 1][0])), np.concatenate((chunk[1], chunks[i + 1][1])))
            del chunks[i]

    #prints the number of chunks
    print(len(chunks))

    #prints the number of chunks that start < 0.5
    print(len([chunk for chunk in chunks if chunk[1][0] < 0.5]))
    #prints the number of chunks that end > 0.75
    print(len([chunk for chunk in chunks if chunk[1][-1] > 0.75]))


    '''PADS THE CHUNKS TO THE LONGEST CHUNK
    #finds the longest chunk 
    maxlen = max([chunk[0].shape[0] for chunk in chunks])
    # pads the chunks to the longest chunk
    for i, chunk in enumerate(chunks):
        if chunk[0].shape[0] < maxlen:
            chunks[i] = (np.pad(chunk[0], ((0, maxlen - chunk[0].shape[0]), (0, 0))), np.pad(chunk[1], ((0, maxlen - chunk[1].shape[0]), (0, 0))))
    '''
        
    pickle.dump(chunks, open("data.pkl", "wb"))

    

if __name__ == '__main__':
    main()
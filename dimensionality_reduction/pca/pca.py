import numpy as np

def pca(input_matrix, dim = 2):
    if dim < 1 or dim > input_matrix.shape[1]:
        raise
    #Step 1
    matrix_mean0 = input_matrix - np.mean(input_matrix,axis=0)
    #Step 2
    samples = matrix_mean0.shape[0]
    covariance_matrix = 1 / ( samples - 1) * np.dot(np.transpose(matrix_mean0), matrix_mean0)
    #Step 3
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    #Step 4, Each row will be an eigenvector in order
    featureVectors = eigenvectors[:,np.argsort(eigenvalues)[::-1]]
    return np.dot(np.transpose(featureVectors[:,:dim]),np.transpose(matrix_mean0))


data = np.array([
        [2.5,2.4],
        [0.5,0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2, 1.6],
        [1, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
        ])

if __name__ == "__main__":
    # print(pca(data))

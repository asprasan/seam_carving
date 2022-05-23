import numpy as np
import imageio
from scipy.signal import convolve2d

def build_energy_mat(gray):
    sobel_filter = np.array([[0.125, 0., -0.125],
        [0.25, 0., -0.25],
        [0.125, 0., -0.125]])
    gray_gradx = convolve2d(gray, sobel_filter, mode='same')
    gray_grady = convolve2d(gray, sobel_filter.T, mode='same')
    energy_mat = (gray_gradx**2 + gray_grady**2)**0.5
    return energy_mat

def build_path_energy_mat(energy_mat):
    # go from top to bottom
    H,W = energy_mat.shape
    for h in range(H-2, -1, -1):
        for w in range(W):          
            energy_mat[h, w] += np.amin(energy_mat[h+1, max(0,w-1):min(W,w+2)])
    return energy_mat

def path_reduce_by_one(energy_mat):
    # first find the smallest energy in first row
    # then find the rest of the path
    path = [np.argmin(energy_mat[0,:])]
    H,W = energy_mat.shape[:2]
    for h in range(1,H):
        min_idx = np.argmin(energy_mat[h,max(0,path[-1]-1):min(W, path[-1]+2)])
        true_idx = np.clip(path[-1] + min_idx - 1, 0, W-1)
        path.append(true_idx)
    return path

def carve_by_one(I, energy_mat, path):
    for k,col in enumerate(path):
        I[k,col:-1] = I[k,col+1:]
        energy_mat[k, col:-1] = energy_mat[k, col+1:]
    return I[:,:-1], energy_mat[:,:-1]

def main(img_path):
    I = imageio.imread(img_path)
    H,W = I.shape[:2]
    # first build the gradient magnitude 
    I_gray = imageio.imread(img_path, pilmode='L')
    energy_mat = build_energy_mat(I_gray)

    # now compute the least energy paths
    energy_mat = build_path_energy_mat(energy_mat)
    # return a list of cols for each row to be removed
    for _ in range(10):
        remove_path = path_reduce_by_one(energy_mat)
        I, energy_mat = carve_by_one(I, energy_mat, remove_path)
    imageio.imwrite('carved_image.jpg', I)


if __name__ == '__main__':
    img_path = 'The_Persistence_of_Memory.jpg'
    main(img_path)
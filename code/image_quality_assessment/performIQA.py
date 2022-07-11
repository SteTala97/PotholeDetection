"""
Created on Sat Jul  2 12:43:23 2022

@author: Stefano Talamona
"""


import os
import cv2 as cv
import numpy as np
import sys
import glob
import collections
from itertools import chain
import pickle
import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from libsvm import svmutil



IMAGE_FORMAT = 'jpg' # ['jpg', 'png', 'JPEG']
DATA_FOLDER = 'original_dataset'
NORM_PATH = r'D:\UNIVERSITA\Magistrale\SecondoAnno\VisualInformationProcessingAndManagement\ProgettoVISUAL\code\image_quality_assessment\normalize.pickle'
SVM_PATH = r'D:\UNIVERSITA\Magistrale\SecondoAnno\VisualInformationProcessingAndManagement\ProgettoVISUAL\code\image_quality_assessment\brisque_svm.txt'
UPPER_BOUND = 190
LOWER_BOUND = -10


def show_img(winname, img):
    cv.namedWindow(winname, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(winname, img)
    k = 0xFF & cv.waitKey()
    if k == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    cv.destroyAllWindows()


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name.lstrip('data/obj/'))
    return names_list


def scale_score(score):
    return (((score - LOWER_BOUND) * 80) / UPPER_BOUND) # maximum score is going to be 80


""" 
The following functions used to perform automatic Image Quality Assessment with BRISQUE features have been taken
from this notebook: https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb
"""
    
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return normalize_kernel(gaussian_kernel)


def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    
    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)


def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })


def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))
    
    coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
    f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)
        
    return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))


def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))
    
    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
    
    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)
    
    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)
    
    return alpha, mean, sigma_l, sigma_r


def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    def calculate_features(coefficients_name, coefficients, accum=np.array([])):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]
        
        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
    
    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    
    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    flatten_features = list(chain.from_iterable(features))
    return np.array(flatten_features, dtype=object)


def plot_histogram(x, label):
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    plt.plot(bins[:-1], n, label=label, marker='o')
    
    
def scale_features(features):
    with open(NORM_PATH, 'rb') as handle:
        scale_params = pickle.load(handle)
    
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
    
    return -1 + (2.0 / (max_ - min_) * (features - min_))


def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model(SVM_PATH)
    scaled_brisque_features = scale_features(brisque_features)
    
    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)



def main():
    cwd = os.getcwd() # Current Working Directory
    cwd = cwd[:-30] # Remove the substring "/code/image_quality_assessment" from the cwd
    generic_path = cwd + '/data/' + DATA_FOLDER 
    im_path = generic_path + '/*.' + IMAGE_FORMAT
    image_names = [file for file in glob.glob(im_path)] # Percorsi delle immagini originali 
    
    scores_list = []
    lower_bound = 100
    upper_bound = 0
    
    for img_name in image_names:
        img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)        
    
        # # Plot MSCN coefficients distribution
        # mscn_coefficients = calculate_mscn_coefficients(img, 7, 7/6)
        # coefficients = calculate_pair_product_coefficients(mscn_coefficients)
        # plt.rcParams["figure.figsize"] = 12, 11
        # for name, coeff in coefficients.items():
        #     plot_histogram(coeff.ravel(), name)
        # plt.axis([-2.5, 2.5, 0, 1.05])
        # plt.legend()
        # plt.show()
        
        brisque_features = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
        downscaled_image = cv.resize(img, None, fx=1/2, fy=1/2, interpolation=cv.INTER_CUBIC)
        downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6) 
        brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
        score = calculate_image_quality_score(brisque_features)
        score = scale_score(score)
        scores_list.append(score)
        if score > upper_bound: upper_bound = score
        if score < lower_bound: lower_bound = score
    
    # print(scores_list)
    print("\n Number of images:", len(scores_list))
    print(" Lower bound:", lower_bound)
    print(" Upper bound:", upper_bound)
    print(" Mean:", sum(scores_list)/len(scores_list))
    print(" Median:", (upper_bound - lower_bound) / len(scores_list))
    
    
    
if __name__ == "__main__":
    main()





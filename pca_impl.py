import os
import numpy
from PIL import Image
import numpy as np


def create_matrix_dataset(filepath):
    dataset = np.empty([1, 4928])
    name_list = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.pgm':
                # resize image to convenient shape
                basewidth = 64
                img = Image.open(os.path.join(root, file))
                img = img.convert('L')
                wpercent = (basewidth / float(img.size[0]))
                hsize = 77
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)

                numpy_data = np.asarray(img).reshape([1, 4928])
                dataset = np.vstack([dataset, numpy_data])
                name_list.append(root.split('/')[-1])

    dataset = np.delete(dataset, 0, 0)
    return dataset, name_list


def normalize_data(data_matrix):
    mean = data_matrix.mean(0)
    data_matrix = np.subtract(data_matrix, mean)
    return data_matrix, mean


def create_covariance_matrix(data_matrix):
    n = data_matrix.shape[0]
    cov_matrix = (1 / (n - 1)) * np.dot(data_matrix.T, data_matrix)
    return cov_matrix


def create_eigenfaces(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    eigen_matrix = np.vstack([eigenvalues, eigenvectors])
    eigen_matrix = eigen_matrix[:, eigen_matrix[0].argsort()[::-1]]
    eigen_matrix = np.delete(eigen_matrix, 0, 0)

    return abs(eigen_matrix)


def transform_images(eigenfaces, normal_data):
    transformed_matrix = np.dot(normal_data, eigenfaces)
    return transformed_matrix


def transform_single_image(filepath, eigenfaces, mean):
    # open and resize image
    basewidth = 64
    img = Image.open(filepath)
    img = img.convert('L')
    wpercent = (basewidth / float(img.size[0]))
    hsize = 77
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    numpy_data = np.asarray(img).reshape([1, 4928])
    normalized_data = numpy_data - mean
    transformed_image = np.dot(normalized_data, eigenfaces)

    return transformed_image


def identify_face(tr_image, tr_matrix, name_list):
    min_dist = numpy.inf
    for i in range(tr_matrix.shape[0]):
        dist = numpy.linalg.norm(tr_image - tr_matrix[i])
        if dist < min_dist:
            min_dist = dist
            name = name_list[i]
    return ' '.join(name.split('_'))


def show_projected_face(tr_image, eigenfaces, pca, mean):
    new_face = np.empty([1, 4928])
    for i in range(len(tr_image)):
        new_face += tr_image[i, 0] * eigenfaces[:, i]

    new_face += mean
    new_face = new_face.reshape([77, 64])
    image = Image.fromarray(new_face)
    image.show()


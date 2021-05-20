import matplotlib.pyplot as plt
from pca_impl import *


def show_images(data_matrix):
    fig, axes = plt.subplots(6, 8, figsize=(20, 15),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(data_matrix[i].reshape(77, 64), cmap='gray')
    plt.show()


def show_eigenfaces(eigenfaces):
    fig, axes = plt.subplots(3, 5, figsize=(20, 15),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(eigenfaces.T[i].reshape(77, 64), cmap='gray')
    plt.show()


def test(filepath, tr_matrix, eigenfaces, mean, name_list):
    correct_count = 0
    count = 0
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.pgm':
                tr_img = transform_single_image(os.path.join(root, file), eigenfaces, mean)
                name = identify_face(tr_img, tr_matrix, name_list)
                if name == root.split('/')[-1]:
                    correct_count += 1
                count += 1

    return correct_count / count


def multi_test(filepath, eigenfaces, mean, name_list, principal_comp):
    results = []
    for pca in principal_comp:
        transform_matrix = transform_images(eigenfaces[:, :pca], normal_data)
        res = test(filepath, transform_matrix, eigenfaces[:, :pca], mean, name_list)
        results.append(res)

    plt.plot(principal_comp, results)
    plt.xlabel('Principal components')
    plt.ylabel('Test set accuracy')
    plt.show()


if __name__ == '__main__':
    filepath = "dataset/train"
    dataset, name_list = create_matrix_dataset(filepath)
    normal_data, m = normalize_data(dataset)
    cov_matrix = create_covariance_matrix(normal_data)
    eigenfaces = create_eigenfaces(cov_matrix)
    transform_matrix = transform_images(eigenfaces[:, :32], normal_data)
    show_eigenfaces(eigenfaces)

# test single image -----------------------------------
    tr_img = transform_single_image('dataset/train/Yurii_Vipshovskyi/1.jpg', eigenfaces[:, :32], m)
    name = identify_face(tr_img, transform_matrix, name_list)
    show_projected_face(tr_img, eigenfaces, 32, m)
    print(name)

# # test on test directory with fixed number of dominating principal components
#     pca = 64
#     filepath2 = "dataset/test"
#     transform_matrix = transform_images(eigenfaces[:, :pca], normal_data)
#     res = test(filepath2, transform_matrix, eigenfaces[:, :pca], m, name_list)
#     print(res)
#
# test on directory with multiple numbers of dominating principal components
#     principal_comp = [6, 12, 20, 32, 45, 50]
#     filepath2 = "dataset/test"
#     multi_test(filepath2, eigenfaces, m, name_list, principal_comp)

from pca_impl import *

if __name__ == '__main__':

    pca = int(input("Type in the number of dominating principal components: "))
    input_image = input("Select image to identify: ")

    # train PCA -------------------------------------------------------
    train_database = "dataset/train"
    dataset, name_list = create_matrix_dataset(train_database)
    normal_data, m = normalize_data(dataset)
    cov_matrix = create_covariance_matrix(normal_data)
    eigenfaces = create_eigenfaces(cov_matrix)
    transform_matrix = transform_images(eigenfaces[:, :pca], normal_data)

    # identify the person
    tr_img = transform_single_image(input_image, eigenfaces, m)
    name = identify_face(tr_img, transform_matrix, name_list)

    print("This is a photo of " + name + "!")

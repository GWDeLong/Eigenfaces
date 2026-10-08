import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Configuration --- 

DATASET_PATH = "./att_faces"
TESTING_PATH = "./testing_images"
IMAGE_SIZE = (112, 92)
NUM_EIGENFACES = 400

# --- Load Images ---

def load_images(dataset_path):
    images = []
    labels = []
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path): 
                img_path = os.path.join(person_path, filename)
                img = Image.open(img_path).convert('L') 
                img = img.resize(IMAGE_SIZE)
                images.append(np.asarray(img, dtype=np.float32).flatten())
                labels.append(person_dir)
    return np.array(images), labels

# --- Compute PCA ---

def compute_eigenfaces(images, num_components):
    scaler = StandardScaler(with_std=False)  
    images_centered = scaler.fit_transform(images)   
    pca = PCA(n_components=num_components,    
              svd_solver='randomized',
              whiten=True)
    pca.fit(images_centered)    
    return pca, scaler.mean_

# --- Plot All Faces

def plot_faces(images, image_shape, num_faces):
    fig, axes = plt.subplots(40, 10, figsize=(10, 40))
    n=0
    for i in range(40):
        for j in range(10):
            axes[i,j].imshow(images[n].reshape(image_shape[1], image_shape[0]), cmap='gray')
            axes[i,j].axis('off')
            n += 1
    plt.show()

# --- Plot Eigenfaces ---

def plot_eigenfaces(pca, image_shape, num_faces=10):
    fig, axes = plt.subplots(1, num_faces, figsize=(15, 5))
    for i in range(num_faces):
        axes[i].imshow(pca.components_[i].reshape(image_shape[1], image_shape[0]), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"PC {i+1}")
    plt.suptitle("Top Eigenfaces")
    plt.show()

# --- Choose Image ---

def choose_image():
    image = -1
    while image not in range(1,40):
        image = input('Select person (1-40): ')
        try:
            image = int(image)
        except ValueError:
            image = -1
    return image

# --- Image Reconstruction ---

def img_reconstruction(user_input):
    fig, axes = plt.subplots(1, 2)
    img_num = (user_input-1)*10
    UTUx = pca.components_.T@(pca.components_@(images[img_num]-mean_face))+mean_face
    axes[0].imshow(images[img_num].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
    axes[1].imshow((UTUx).reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
    axes[0].set_title(f"Image {labels[img_num]}")
    axes[1].set_title(f"Reconstructed {labels[img_num]}")
    for i in range(2):
        axes[i].axis('off')
    plt.show()    
    
# --- Image Reconstruction W/ Steps

def img_reconstruction_w_steps(user_input):
    fig, axes = plt.subplots(1, 8)
    img_num = (user_input-1)*10
    i = 0
    for r in [1, 5, 10, 25, 50, 100, 200, 400]:
        x = images[img_num]
        UTUx = pca.components_[0:r].T@(pca.components_[0:r]@(x - mean_face))+mean_face
        axes[i].imshow(UTUx.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
        axes[i].set_title(f"r = {r}")
        axes[i].axis('off')
        i += 1
    plt.show()
    
# --- Image Classification ---

def img_classify(user_input1, user_input2):   
    Person1 = user_input1-1
    Person2 = user_input2-1
    P1 = images[Person1*10:Person1*10+10]
    P2 = images[Person2*10:Person2*10+10]
    P1 = P1 - np.tile(mean_face,(10,1))
    P2 = P2 - np.tile(mean_face,(10,1))
    PCAmodes = [6,7]
    PCACoordsP1 = pca.components_.T[:,PCAmodes - np.ones_like(PCAmodes)].T@P1.T
    PCACoordsP2 = pca.components_.T[:,PCAmodes - np.ones_like(PCAmodes)].T@P2.T
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(images[Person1*10].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray'), axes[0].axis('off')
    axes[1].imshow(images[Person2*10].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray'), axes[1].axis('off')
    plt.show()
    plt.plot(PCACoordsP1[0,:],PCACoordsP1[1,:],'d',color='k',label=f'Person {Person1+1}')
    plt.plot(PCACoordsP2[0,:],PCACoordsP2[1,:],'^',color='r',label=f'Person {Person2+1}')
    plt.legend()
    plt.show()

# --- Face Matching ---
    
def face_match():
    plt.imshow(test_images[0].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
    plt.axis('off'), plt.show()
    diff = []
    for n in range(len(images)):
        diff.append(mean_squared_error(test_images[0],images[n]))
    people_diff = []
    for n in range(40):
        people_diff.append(np.average(diff[n*10:n*10+10]))
    plt.bar(range(400),diff),plt.show()
    plt.bar(range(40),people_diff),plt.show()
    matching = people_diff.index(min(people_diff))
    print(f'Best match is person {labels[matching]}')
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_images[0].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
    axes[1].imshow(images[matching].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
    axes[0].set_title(f"Input Image")
    axes[1].set_title(f"Best Match {labels[matching]}")
    for i in range(2):
        axes[i].axis('off')
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    print("Loading images...")
    images, labels = load_images(DATASET_PATH)
    test_images, test_labels = load_images(TESTING_PATH)
    print(f"Loaded {len(images)} images.")
    print("Computing PCA...")
    pca, mean_face = compute_eigenfaces(images, NUM_EIGENFACES)
    print("Done.")
    user_input = ''
    while user_input != '7':
        print("""
Make Selection:

1. Show all faces
2. Show Eigenfaces
3. Show mean face
4. Image Reconstructor
5. Image Classifier
6. Image Matcher
7. Exit Program

""")
        user_input = input()
        if user_input == '1':
            plot_faces(images, IMAGE_SIZE, len(images))
        elif user_input == '2':
            plot_eigenfaces(pca, IMAGE_SIZE, num_faces=10)
        elif user_input == '3':
            plt.imshow(mean_face.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray')
            plt.title("Mean Face")
            plt.axis('off')
            plt.show()
        elif user_input == '4':
            print('Person 1:')
            Im1 = choose_image()
            img_reconstruction(Im1)
            img_reconstruction_w_steps(Im1)
        elif user_input == '5':
            print('Person 1:')
            Im1 = choose_image()
            print('Person 2:')
            Im2 = choose_image()
            img_classify(Im1, Im2)
        elif user_input == '6':
            face_match()
        input('Press ENTER to continue...')
    print('Program successfully exited.')
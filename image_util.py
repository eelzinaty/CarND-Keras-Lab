import cv2
import numpy as np

def translateImage(image):
    t_x = (np.random.randn(1)*.5)[0]
    t_y = (np.random.randn(1)*.5)[0]
    #print(t_x,t_y)
    rows,cols,_ = image.shape
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def rotateImage(image):
    theta = (np.random.randn(1)*5)[0]
    #print(theta)
    rows,cols,_ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def generateAugmentedImages(X_data,y_data):
    num_images = 5 * len(X_data)
    x_aug = np.zeros(shape=(num_images,32,32,3)).astype(np.uint8)
    y_aug = np.zeros(shape=(num_images,n_classes))
    aug_idx = 0
    for idx, img in enumerate(X_data):
        lbl = y_data[idx]
        for itr in range(5):
            new_image = translateImage(rotateImage(img))
            x_aug[aug_idx] = new_image
            y_aug[aug_idx] = lbl
            aug_idx = aug_idx + 1
    X_data = np.concatenate((X_data, x_aug), axis=0)
    y_data = np.concatenate((y_data, y_aug), axis=0)
    #print(y_data)
    return X_data, y_data

### convert to YUV, extract Y and normalize it

def imagestoY(batch_x, batch_y,n_classes):
    num_images = len(batch_y)
    b_x = np.zeros(shape=(num_images,32,32,1)).astype(np.uint8)
    b_y = np.zeros(shape=(num_images,n_classes)).astype('float32')
    idx = 0
    for x in batch_x:
        #plt.figure(figsize=(1,1))
        #plt.imshow(x)
        img_out = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
        
        #plt.figure(figsize=(1,1))
        #plt.imshow(img_out)
        img_out = img_out[:,:,0].reshape(32,32,1)
        img_out = img_out - np.mean(img_out)
        b_x[idx] = img_out
        b_y[idx] = batch_y[idx]
        
    return b_x, b_y

#x,y = imagestoY(X_train[0:2],y_one_hot[0:2])

### convert to Gray sacle, then normalize using CLAHE.
### http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

def imagetoCLAHE(batch_x, batch_y,n_classes):
    num_images = len(batch_y)
    b_x = np.zeros(shape=(num_images,32,32,1)).astype(np.uint8)
    b_y = np.zeros(shape=(num_images,n_classes)).astype('float32')
    idx = 0
    for x in batch_x:
        img_grey = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img_grey)
        #cl1 = cl1.reshape(32,32,1)
        b_x[idx] = cl1.reshape(32,32,1)
        b_y[idx] = batch_y[idx]
    return b_x,b_y
#x,y = imagetoCLAHE(X_train[0:2],y_one_hot[0:2])

def imagetoCLAHE_X(batch_x):
    num_images = len(batch_x)
    b_x = np.zeros(shape=(num_images,32,32,1)).astype(np.uint8)
    idx = 0
    for x in batch_x:
        img_grey = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img_grey)
        b_x[idx] = cl1.reshape(32,32,1)
    return b_x

### Python generator function that applied to fetch batches and do data preprocessing        
# https://github.com/justheuristic/prefetch_generator
def iterate_minibatches(num_examples, batch_size, X_data, y_data, is_training = 1):
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        if is_training == 1:
            batch_x, batch_y = generateAugmentedImages(batch_x, batch_y)
            #batch_x, batch_y = shuffle(batch_x, batch_y)
        #batch_x = imagestoY(batch_x)
        #batch_x , batch_y= imagetoCLAHE(batch_x,batch_y)
        yield batch_x,batch_y

        
def iterate_gen(num_examples, batch_size, X_data, y_data, is_training = 1):
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        if is_training == 1:
            batch_x, batch_y = generateAugmentedImages(batch_x, batch_y)
            batch_x, batch_y = shuffle(batch_x, batch_y)
        #batch_x = imagestoY(batch_x)
        batch_x, batch_y = imagetoCLAHE(batch_x,batch_y)
        #print(batch_y[0])
        yield batch_x,batch_y
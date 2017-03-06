# CarND-Vehicle-Detection
The goal of the project is to write a software pipeline to detect vehicles in a video using sliding window classification approach.
Udacity provides two sets of 64x64 images for binary classification:
* [8792 images of vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* [8968 images of non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

The vehicle detection pipeline should correctly track vehicles on the road and eliminate false positive detections, which can be dangerous for a real Self-Driving Car for several reasons.

Here I describe the most  important  parts of the code according to Udacity's project rubric:
* Histogram of Oriented Gradients (HOG) usage
* Combined Features
* Classifier Training
* Sliding Window Search implementation
* False Positive Removal
* Video processing

My personal goal was to implement a fast pipeline, therefore I have to balance vehicle detection versus speed.

[//]: # (Image References)
[image1]: ./output_images/Hog_Features.png
[image2]: ./output_images/hotwindow_example.png
[image3]: ./output_images/heatmap.jpg
[image4]: ./output_images/cardetection.jpg
[image5]: ./output_images/u-net-architecture.png
[image6]: ./output_images/example_label_mask1.png
[image7]: ./output_images/generator_output.png
[image8]: ./output_images/result-gnet1.png
[image9]: ./output_images/result-gnet2.png
[video1]: ./project_video_solution.mp4


## Histogram of Oriented Gradients (HOG)
The heart of the pipeline is Histogram of Oriented Gradients or HOG, which is implemented in scikit image package as [`skimage.feature.hog`](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog).

Here is a wrapper code snippet used in pipeline:
```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

Essential parameters here are `orient`, `pix_per_cell` and `cell_per_block` which have to be defined to work well for this task.

Another tricky thing is `img` -- this must be 1-channel input and this can be either grayscaled image or a particular color channel from a color space.

Here is an example for one vehicle image with its hog features and a non vehicle image with its hog features:
![alt text][image1]

## Combined Features

I used the function of the Udacity lesson to use HOG features, spatial features and color histogram features in combination. I adopted the function to process only a single image. Here is the code snippet:

```
def extract_features(image,  color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)

    # 7) Compute HOG features if flag is set
    if hog_feat:

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

```
I experimented with different parameter sets. One important point  is that extracting the Hog features is quite slow.  Extracting the hog features for only one color channel is three times faster as extracting the Hog features for all three color channels, which results in a much faster vehicle detection pipeline. The disadvantage is that a SVC classifier loose more than 2% test accuracy, which results in much more false positives. I find out, that using the MLPC classifier in sckit learn results not only in a higher test accuracy but also that this classifier loose only 0.2% test accuracy using only one color channel for Hog features.

I implemented a function which extracts the combined features for an image where all the parameters are set:

```
def single_img_features(image):
    # Standartize images to be uint8 data type
    if isinstance(image[0][0][0], np.float32):
        image = np.uint8(image * 255)

    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

    # after a lot of experiments these parameter settings perform quite well
    spatial_size = (32, 32)  # Spatial binning dimensions
    spatial_feat = True  # Spatial features on or off

    hist_bins = 64  # Number of histogram bins
    hist_feat = True  # Histogram features on or off

    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel =0 # #"ALL"  # Can be 0, 1, 2, or "ALL"

    hog_feat = True  # HOG features on or off

    return extract_features(image, color_space=color_space,
                               spatial_size=spatial_size, hist_bins=hist_bins,
                               orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block,
                               hog_channel=hog_channel, spatial_feat=spatial_feat,
                               hist_feat=hist_feat, hog_feat=hog_feat)

```

This parameter settings performs quite well. Most important to note is that YCrCb colorspace is used and color channel 0 is used for the Hog feature extraction.

## Training the classifier

### Feature normalization and creating training and test data sets.

It is important to do a feature normalization. This is done with the following code snippet:

```
# use standardscaler, which removing the mean and scaling to unit variance
scaler = StandardScaler().fit(X)

# Apply the scaler to the data set
scaled_X = scaler.transform(X)
```

Then I created training and test sets with the following code snippet:

```


# shuffle the training set
scaled_X, y = shuffle( scaled_X, y, random_state=42)
# split into train, validation and test sets
X_train, X_test, y_train, y_test= train_test_split(scaled_X, y, test_size=0.2, random_state=1234)
```
Size of the training set is 14208, size of the test set 3552.

### Training a SVC classifier

Training of a linear SVC classifer was done with the following code snippet:

```
svc = LinearSVC()
clf=svc.fit(X_train, y_train)
# Check the score of the SVC
print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
%time print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
```

The results are:
Train Accuracy of SVC =  1.0
Test Accuracy of SVC =  0.988175675676
Wall time: 90.6 ms

### Training a MLPC classifier

Training of a linear MLPC ( Multi-layer Perceptron,http://scikit-learn.org/stable/modules/neural_networks_supervised.html )  classifer was done with the following code snippet:

```
from sklearn.neural_network  import MLPClassifier

mlpc=MLPClassifier(hidden_layer_sizes =(40,40))
clf2=mlpc.fit(X_train, y_train)
# Check the score of the SVC
print('Train Accuracy of MLPC = ', mlpc.score(X_train, y_train))
%time print('Test Accuracy of MLPC = ', mlpc.score(X_test, y_test))

```

Train Accuracy of MLPC =  1.0
Test Accuracy of MLPC =  0.994369369369
Wall time: 106 ms

The results are much better than with a SVC classifier and the processing time are very similar. The size of the hidden layers where optimized by some trial and error.
I used this classifier in my detection pipeline instead of a SVC classifier.


### Sliding Window Search implementation

I used the code from the Udacity lessons to implement the sliding window search:

```
# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(image, windows, clf, scaler):  
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            prob=clf.predict_proba(test_features)
            if prob[0][1]>0.55:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

```

I implement then a function to detect  all hot windows in an image:

```
# define a function to find boxes with cars
# Cache for windows

def find_hot_windows(image):
    windows_all=[]   
    #after a lot of trials this works quite well and very fast
    img_sizes=[90,100,116,140,164]
    y_stops=[500,500,540,580,580]

    for i in range(0,len( img_sizes)):
        size=img_sizes[i]
        windows = slide_window(image, x_start_stop=[None,None] , y_start_stop=[400,600] ,  
                            xy_window=(size, size), xy_overlap=(0.5,0.5))
        windows_all+=windows

    hot_windows=search_windows(image,windows_all,clf2,scaler)

    return hot_windows
```
The parameters img_sizes. y_start_stop where found doing much experimentation. My goal was to detect all cars which are not too far away and to realize a fast detection pipeline.  Therefore the image sizes are quite large and the y range is quite restricted. The following image shows the result for some of the test images:

![alt text][image2]
There are some false positives but the cars are well detected.


## False Positive Removal

I used the code from the Udacity lesson using a heatmap for false positive removal:

```
# create a heat map
def add_heat(heatmap, boxlist):
    # Iterate through list of bboxes    
    for box in boxlist:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

# Applies threshold to the heatmap, zeroing out anything below the threshold
def apply_threshold(heatmap, threshold):
    new_heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    new_heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return new_heatmap

    # Draws bounding boxes over an image
    def draw_labeled_bboxes(image, labels):
        # Iterate through all detected cars
        img = np.copy(image)
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img    

```

The following image shows the result for the fourth image above:

![alt text][image3]

It works quite well leaving only one false positive on the left side.

##Video processing
For video processing and removal of false positives I implemented the following function:

```

def find_cars_in_image(img,threshold=0.9, prev_hot_windows=None):
     # Define global variables
    hot_windows=find_hot_windows(img)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)    
    heatmap = add_heat(heat, hot_windows)


    # Heat threshold to filter false positives
    heat_threshold = threshold
    look_back_count = 5
      # If we a processing a video, we might have hot windows information from previous frames

    if prev_hot_windows is not None:

        for cur_hot_windows in prev_hot_windows:
            current_heatmap=np.zeros_like(img[:,:,0]).astype(np.float)   
            add_heat(current_heatmap, cur_hot_windows)
            heatmap += current_heatmap
        if len(prev_hot_windows) > look_back_count:
            prev_hot_windows.popleft()

    heatmap_thresholded = apply_threshold(heatmap, heat_threshold)    
    labels = label(heatmap_thresholded)
    window_img = draw_labeled_bboxes(img, labels)
    return window_img, prev_hot_windows
```

First the how windows are detected. Then with a heat map of the last 5 images the false positives are removed. The following image shows the results for some of the test images (here are no previous slides):
![alt text][image4]

The final function for video processing is:
```
from collections import deque
# Processes one frame of video
def process_frame(image):
    # Obtain previous hot frames
    global prev_hot_windows
    prev_hot_windows = deque([])
    # Find cars in given frame and draw bounding boxes around them
    result_image, prev_hot_windows = find_cars_in_image(image,0.2, prev_hot_windows)

    # Return resulting image
    return result_image
```
The  resulting project videos generated by described pipeline can be found in this repository, for the project_video the result is project_video_solution.mp4


There are no false positives only cars on the other lane are detected sometimes. The overall pipeline is quite fast, on my laptop with I7 core I achieved about 3.5 frames per second. That is not real time, but there are no special code optimizations for Hog feature extraction.

###Discussion


The optimization of several parameters - feature extraction and very important windows sizes are very critical to achieve sucess. The boundary between false negatives and false positives is very small and that indicates that this hand tuned classical image processing appproach is not very robust.  Another issues is, that the processing pipeline is not fast enough, but hog feature extraction can be made much faster, if it is applied to the lower half of an image at once. Further performance improvement can be achieved by using a GPU for the image processing routines.

As in other image processing areas, things could be much improved and made more robust by using deep neural nets. One option is using a VGG like network which has an regression head for the localization of objects and and classification head for object classification (car, truck, bike, bicyle, predistrian, child..) . Another approach would be using an UNetto do image segmentation to localize objects.  I currently are experimenting with an UNet for car detection, but at the moment there a too much false positives but I m working on it. On a GTX 1080 GPU 10 to 15 frames per second can be processed. For the training of the UNet  I m using the Udacity annotated driving data set (https://github.com/udacity/self-driving-car/tree/master/annotations).  I will work on it again when I pass this submission and then I will add the code and result videos to this repo too.

# --------------------------------------------------------------------------------------
# U-Net for vehicle detection

## This is the notebook with the training code



## Overview

In this notebook , we will implement and train an  U-net for detecting vehicles in a video stream of images provided by Udacity. U-net is a encoder-decoder type of network for pixel-wise predictions. UNet are special Convets: receptive fields after convolution are concatenated with the receptive fields in up-convolving process. This allows the network to use features from lower layers and features from up-convolution. This up-convolution makes training harder in the sense that much more memory is required as in standard Conv-Nets where only downconvolution is done.  U-nets are used extensively for biomedical applications to detect cancer, kidney pathologies and tracking cells and so on. U-net has proven to be very powerful segmentation tool.

![alt text][image5]
U-net, taken from http://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/


The input to U-net is a resized 960X640 3-channel RGB image and output is 960X640 1-channel mask of predictions. We wanted the predictions to reflect probability of a pixel being a vehicle or not, so we used an activation function of sigmoid on the last layer.

The solution in this notebook is based in the original research paper on [U-net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and the prize winning submission to kaggle’s ultrasound segmentation challenge. The UNet-Code is based on this repository https://github.com/orobix/retina-unet


## The Data

We used annotated vehicle data set provided by [Udacity](https://www.udacity.com/). The [4.5 GB data set](https://github.com/udacity/self-driving-car/tree/master/annotation) was composed of frames collected from two of videos while driving the Udacity car around Mountain View area in heavy traffic. The data set contained a label file with bounding boxes marking other cars, trucks and pedestrians. The entire data set was comprised of about 22000 images. We combined cars and trucks into one class vehicle, and dropped all the bounding boxes for pedestrians. For each image a set of bounding boxes is provided.



## Data preparation and augmentation

Frames were obtained from a video feed, therefor shuffling is very important.  Data is splited into training and testing data sets, 2000 images are used for testing. Data augmentation on training data:

- translation to account for cars beeing at different locations
- brightness to account for differenct lightning conditions
- stretching


## Training
Goal is to train the UNet to identify the bounding boxes of the cars in the image. That means that it shall learn to generate a mask which highlights all cars in an image. I choose image sizes of 640x980 and 480x720 for training. For the large image size I trained on a titan X pascal GPU with a batch size of 8 (larger batch sizes raise an out of memory error) and the the smaller image size I trained on a GTX1080 GPU with a batch size of 8. In both cases the training time for 1000 epochs where about 160s and 106 s. I trained for several hundred epochs. I used Keras with tensorflow backed and used approximate Intersection over Union (IoU) between the network output and target mask as objective function.

## Remarks

As in  many cases preparing the data and training a conv net is quite straighforward. The only difference here is the special UNet structure and that we train a deep network to do image segmentation and not image classification.  Compared to the complicated feature engineering with color histograms, HOG, and the sliding windowing technique with the classical image processing approach in the CarND P5 project it is fairly simple and no tuning and so on is required this stage! We will see in the notebook which does the car dectection that using the pretrained Unet model it is really very simple and powerful and very robust.

## Additional links:

1. U-Net: Convolutional Networks for Biomedical Image Segmentation: http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
2. Good collection of various segmentation models: https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
3. Original prize winning submission to Kaggle https://github.com/jocicmarko/ultrasound-nerve-segmentation

## Target set preparation
We used the bounding boxes provided as masks fpr defining cars. The left images below shows real scences with bounding boxes around the cars and the right images shows the image masks we want to predict.

![alt text][image6]

Furthermore we use data augmentation to increase the training data set and enable the network to generalize more. We use the following data augmentation techniques:
- stretching
- translation
- Brightness augmenation

## Model

 We will use   U-nets for detecting vehicles in a video stream of images provided by Udacity. U-net is a encoder-decoder type of network for pixel-wise predictions. UNet are special Convets: receptive fields after convolution are concatenated with the receptive fields in up-convolving process. This allows the network to use features from lower layers and features from up-convolution. This up-convolution makes training harder in the sense that much more memory is required as in standard Conv-Nets where only downconvolution is done.  U-nets are used extensively for biomedical applications to detect cancer, kidney pathologies and tracking cells and so on. U-net has proven to be very powerful segmentation tool.

 We use Keras for the implementation of U-Nets.

 The first U-Net model is a slightly adopted model based  on repository https://github.com/orobix/retina-une. It was adopted a bit to save GPU memory:

 ```
 def get_gnet(drop=0.0):
     inputs = Input((img_rows, img_cols,3))
     inputs_norm = Lambda(lambda x: x/127.5 - 1.)
   #  conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)  
     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)  
     conv1 = Dropout(drop)(conv1)
   #  conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
    # up1 = UpSampling2D(size=(2, 2))(conv1)

     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
     conv2 = Dropout(drop)(conv2)
     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
     #
     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
     conv3 = Dropout(drop)(conv3)
     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
     #
     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
     conv4 = Dropout(drop)(conv4)
     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
     #
     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
     conv5 = Dropout(drop)(conv5)
     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
     #
     up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
     conv6 = Dropout(drop)(conv6)
     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
     #
     up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
     conv7 = Dropout(drop)(conv7)
     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
     #
     up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
     conv8 = Dropout(drop)(conv8)
     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
     #
   #  pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

   #  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
     conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv8)
     conv9 = Dropout(drop)(conv9)
  #   conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
     conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

     #
     conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

     model = Model(input=inputs, output=conv10)


     return model

 ```


 The second is a bit more adopted to save even more GPU memory:

 ```
 def get_adopted_unet():
     inputs = Input((img_rows, img_cols,3))
     inputs_norm = Lambda(lambda x: x/127.5 - 1.)
     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1)
     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

     up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

     up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

     up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)

     up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
     conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up9)
     conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

     conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

     model = Model(input=inputs, output=conv10)


     return model

 ```

# Objective Function
e defined a custom objective function in keras to compute approximate Intersection over Union (IoU) between the network output and target mask. IoU is a popular metric of choice for tasks involving bounding boxes. The objective was to maximize IoU, as IoU always varies between 0 and 1, we simply chose to minimize the negative of IoU.
Intersection over Union (IoU) metric for bounding boxes

Instead of implementing a direct computation for intersection over union or cross entropy, we used a much simpler metric for area where we multiply two times the network’s output with the target mask, and divide it by the sum of all values in the predicted output and the true mask.

```
##### Image size,
img_rows = 640
img_cols = 960

### IOU  coeff and loss calculation
smooth=1.0
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)
```

For data input generation for the U-Net we use a data generator:
```
#training and test generators using augmentation
def generate_train_batch(data,batch_size = 32):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            name,img,bb_boxes = load_image_name(df_vehicles,i_line,size=(img_cols, img_rows),
                                                  augmentation=True, trans_range=50, scale_range=50)
            img_mask = get_mask_segmentation(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks


def generate_test_batch(data,batch_size = 32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line+len(data)-2000
            name,img,bb_boxes = load_image_name(df_vehicles,i_line, size=(img_cols, img_rows),
                                                  augmentation=False,  trans_range=0, scale_range=0 )
            img_mask = get_mask_segmentation(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks
```

The following image shows an example output of the generator. Left are the augmentated images, in the mid the image masks and at the right side the result of applying the image masks.
![alt text][image7]


## Training        
Training was done on a GPU server with Titan X Pascal GPU with 12GB RAM. For the first U-Net model only a batch size of 8 was possible without running out of GPU memory, for the second U-Net model a batch size of 16 was possible.  For own experiments you need at least a fast GPU with 8 GB RAM (12 or 16GB are much better!). Both models were trained for 100, 200, 300, 400 and 500 epochs with a sample size of 1000 per epoch. After each 100 epochs the model and weights was saved.

## Results first model after 500 epochs
The image below shows some test results for the first model. On the left side the original images is show, in the middle the ground truth boxes and on the right side the predicted segmentation masks. The accuracy is quite high.
![alt text][image8]

We also applied the network on some of the Udacity test images. The image below shows an example with a false positive but both cars are correctly recognized:

![alt text][image9]

## Results or the second model after 500 epochs
The results are quite similar without a clear winner.  The second model however consumes lesser GPU memory.

## Processing time for test data
Processing time was about 450 ms for 20 images which includes loading from disk- This is about 22 ms per image which is quite fast-
## Notebook
Code for the training of the U-Nets is in file UNet-Vehicle-Detection-Train.ipynb

## Vehicle Detection with U-Nets
Code is in UNet-Vehicle-Detection.ipynb.

The processing pipeline for detection of cars is suprisingly simple:

```
def process_frame(image):   

    image_bb = np.copy(image)
    bbox_cars = get_BB_new(image_bb)
    result=image_bb
    img_res_shape = result.shape
    for bbox in bbox_cars:
        cv2.rectangle(result,
                      (np.int32(bbox[0][0]*img_res_shape[1]/img_cols),
                       np.int32(bbox[0][1]*img_res_shape[0]/img_rows)),
                      (np.int32(bbox[1][0]*img_res_shape[1]/img_cols),
                       np.int32(bbox[1][1]*img_res_shape[0]/img_rows)),(0,255,0),6)
    return result
```

Most work is done in the function get_BB_new:

```

def pred_for_img(img):
    img = cv2.resize(img,(img_cols, img_rows))
    img = np.reshape(img,(1,img_rows, img_cols,3))
    pred = model.predict(img)
    return pred,img[0]


def get_labeled_bboxes(img, labels):
    # Get labeled boxex
    bbox_all = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        if ((np.max(nonzeroy)-np.min(nonzeroy)> 40) & (np.max(nonzerox)-np.min(nonzerox)> 40)):
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image       
            #cv2.rectangle(img, bbox[0], bbox[1], (0,0,255),6)
            bbox_all.append(bbox)
    # Return the image
    return bbox_all

def get_BB_new(img):
    # Take in RGB image
    pred,img = pred_for_img(img)
    img  = np.array(img,dtype= np.uint8)
    img_pred = np.array(255*pred[0],dtype=np.uint8)
    heatmap = img_pred[:,:,0]
    heatmap = smooth_heatmap(heatmap)
    #print(np.max(heatmap))
    heatmap[heatmap> 240] = 255
    heatmap[heatmap<=240] = 0    
    labels = label(heatmap)

    bbox_all = get_labeled_bboxes(np.copy(img), labels)
    return bbox_all
```

First the image is segmented using the function pred_for_image. The false positives are filtered out by averaging the output of the U-Net for the last N frames and thresholding them at 240 pixel intensity, further all bounding boxes widths or heights below 40 pixels were removed.

On a titan X a frame rate of about 20 frames per second was achieved to produce the resulting movies.

# CarND-Vehicle-Detection
The goal of the project is to write a software pipeline to detect vehicles in a video using sliding window classification approach.
Udacity provides two sets of 64x64 images for binary classification:
* [8792 images of vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* [9666 images of non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

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
# Obtains a feature vector using preselected set of parameters
def single_img_features(image):
    # Standartize images to be uint8 data type
    if isinstance(image[0][0][0], np.float32):
        image = np.uint8(image * 255)

    color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

    # after a lot of experiments these parameter settings perform quite well
    spatial_size = (32, 32)  # Spatial binning dimensions
    spatial_feat = True  # Spatial features on or off

    hist_bins = 64  # Number of histogram bins
    hist_feat = True  # Histogram features on or off

    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 1 #"ALL"  # Can be 0, 1, 2, or "ALL"
    hog_feat = True  # HOG features on or off

    return extract_features(image, color_space=color_space,
                               spatial_size=spatial_size, hist_bins=hist_bins,
                               orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block,
                               hog_channel=hog_channel, spatial_feat=spatial_feat,
                               hist_feat=hist_feat, hog_feat=hog_feat)

```

This parameter settings performs quite well. Most important to note is that YUV colorspace is used and color channel 1 is used for the Hog feature extraction.

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
Test Accuracy of SVC =  0.977477477477
Wall time: 81.6 ms

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
Test Accuracy of MLPC =  0.991835585586
Wall time: 98.1 ms

The results are much better than with a SVC classifier and the processing time are very similar. The size of the hidden layers where optimized by some trial and error.


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
            #7) If positive (prediction == 1) then save the window
            if prediction ==1:
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
    img_sizes=[96,108,120,132]

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
There are some false positives but the cars are well detected with the exception of the car in the third image which is too far away.


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
 #   heatmap = np.sqrt(heatmap)

    # Heat threshold to filter false positives
    heat_threshold = threshold
    look_back_count = 10
      # If we a processing a video, we might have hot windows information from previous frames
    if prev_hot_windows is not None:

        for frame_hot_windows in prev_hot_windows:
            for window in frame_hot_windows:
                heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
        if len(prev_hot_windows) > look_back_count:
            prev_hot_windows.popleft()
    heatmap_thresholded = apply_threshold(heatmap, heat_threshold)    
    labels = label(heatmap_thresholded)
    window_img = draw_labeled_bboxes(img, labels)
    return window_img, prev_hot_windows
```

First the how windows are detected. Then with a heat map of the last 10 images the false positives are removed. The following image shows the results for some of the test images (here are no previous slides):
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

There are no false positives only cars on the other lane are detected sometimes. The overall pipeline is quite fast, on my laptop with I7 core I achieved about 2.5 frames per second. That is not real time, but there are no special code optimizations for Hog feature extraction.

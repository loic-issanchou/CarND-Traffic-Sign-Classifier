# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

NOTE : I can not open the writeup_template correctly on my PC with Jupyter. So i'm not able to show images in this writeup. Please, see my notebook to visualize images result. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. it are bars charts showing how many images correspond to each signs for the training, validation and test data set.

See my notebook please.
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale in order to overcome lighting conditions and color.

Here is an example of a traffic sign image before and after grayscaling.

See my notebook please
![alt text][image2]

As a last step, I normalized the image data in order to obtain a mean zero, or almost, and equal variance.

Here is an example of an original image and an augmented image:

See my notebook please, 4th cell.
![alt text][image3]

The difference between the original data set and the augmented data set is that the augmented data set have now less data and so training time would be reduce. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten		      	| outputs 400 									|
| Dropout				| keep_prob=0.5 training set; =1.0 test set    	|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				| keep_prob=0.5 training set; =1.0 test set    	|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Dropout				| keep_prob=0.5 training set; =1.0 test set    	|
| Fully connected		| outputs 43  									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 100. This allows to have almost all or all of the signs at each epoch. More than 100 will require, i think, more epoch to reach the goal of an accuracy at 0.93.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.0 %
* validation set accuracy of 96.4 % 
* test set accuracy of 94.9 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? Lenet architecture was chosen because it is relatively simple to implement and it works well.
* What were some problems with the initial architecture? With Lenet architecture, we are limited in number of data and training time is longer than an AlexNet architecture. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I have add dropout to the Lenet architecture according to the indications of the following website : http://tf-lenet.readthedocs.io/en/latest/tutorial/dropout_layer.html
* Which parameters were tuned? How were they adjusted and why?
"Keep probability" was tuned according to some publications on CNN which preconize 0.5 for training and 1.0 for validation and test set.
Also, "batch_size" was tuned like i explained upper and the number of epoch was increased in order to obtain more accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Dropout helped me to increase accuracy according to avoid over-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

See my notebook please.
![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

There is no particular difficulty to classify these images. They are clear and in good light conditions.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| Keep Right    								| 
| Turn left ahead		| Turn left ahead 								|
| General Caution		| General Caution								|
| Road work	      		| Road work						 				|
| Speed limit (60km/h)	| Speed limit (60km/h)							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model is sure that this is a Keep Right sign (probability of 1.0), and the image does contain a Keep Right. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep Right   									| 
| 1.27277814e-25		| Turn left ahead								|
| 5.34431963e-35		| No entry										|


For the second image, the model is not very sure that this is a Turn left ahead sign (probability of 0.49), but the image contain effectively a Turn left ahead sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Turn left ahead   							| 
| .27     				| Keep Right 									|
| .07					| Speed limit (60km/h)							|

For image three, the model is sure that this is a General Caution sign (probability of 0.99), and the image does contain a General Caution sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General Caution								| 
| 5.25890528e-06		| Traffic signals								|
| 1.38871243e-08 		| Pedestrians									|

For image for, the model is relatively sure that this is a Road work sign (probability of 0.94), and the image does contain a Road work sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94        			| Road work   									| 
| .03     				| Bumpy road									|
| .02					| Bicycles crossing								|

For the second image, the model is sure that this is a Speed limit (60km/h) sign (probability of 0.99), and the image does contain a Speed limit (60km/h) sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (60km/h)							| 
| .0008     			| Speed limit (80km/h) 							|
| .00002				| Speed limit (50km/h)							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


```


      File "<ipython-input-1-7d719b008c42>", line 7
        ---
           ^
    SyntaxError: invalid syntax
    


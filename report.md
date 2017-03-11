# **Traffic Sign Recognition** 
## Link to the [project code](https://github.com/kpasad/Traffic-Sign-Classifier/)

### Data Set Summary & Exploration
The data set contains 34799 training sample, of 32x32 RGB images (3 channels). There are 12630 test images. There are 43 hypothesis for classification. Some observations:

* The traffic signs are pre-processed from their original size as available in [master data set] (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), to a 32x32 pixel images 
* Images are tightly cropped and centered. There is no ambiguity in location of the signs within the images. This is very important as will be discussed in context of external data
* Data set is unbalanced. 
* Running cell #3 in the notebook will generate one random instances of all signs.  
![Random instance of every sign](/images/diff_signs.png) 
* Running cell #4, generates diffrents instance of a single sign
![Random instances of one sign](/images/diff_imgs_same_sign.png) 
* From the small sample, it seems that variations in perspective and lighting condition (particularly contrast) are the largest. There is rotation but not much of shift.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
My intial approch was to begin with the simplest model and incrementally add addtional information.
I began with gray scale non normalised images. After getting an intial benchmark, I tried a mean normalised RBG image. See cell #7
How to perform Z-normalisation is not very clear in a RGB image. Since the statistics of each channel are diffrent, a single variance normalisation across all three channels is not the right approch.
I did not try it. Instead I tried UVY channel conversion based on recommendation of the paper, and using only one channel. In a  UVY image the V and Y channels contain the color information.
But a simple visual inspection revealed that it was difficult to discriminate the images based on single color channel. Training the LeNet model corrobated the visual inspection with 
a significantly poor performance.
I did not do any additional processing. I added Z normalisation and compared it with min-max normalisation on grayscale images. Both were roughly the same with a short triainign window
of 20 epoch. I went with variance normalisation. 
In conculsion, for this data set, color does not provide any discriminative features.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Initially I used the classical train-test split. The performance on this validation was extremely good, approching over 99% accuracy, even on plain vanilla LeNet model.
However, the validation set provided with the project results in a 95% accuracy floor. This led me to belive that the validation data is statistically diffrent from the training set. 
I stuck to the validation data provided and tried to generalise the network for this validation data. I used the test data as a absolute metric, and did not attempt to improve test score as I considered it as the 'true' unseen data.


In order to provide additional data, I created augmentations. The number of augmented images were calculated to create a balanced data. e.g. A sign with fewer sample has more augmented images. In retrospect, this was probably not a good idea. The network will generalize by a varying degree to diffrent signs. 
On the first iteration I seperated the various augmemtations, applying one augment at a time e.g. eighther of the following:
x-shift, y shift, rotation. I did not add perspective distortion, though it seems the most likely distortion. To approximate the perspective distortion, later  I added shearing. I was a 
bit conservative in augmentation parameters since the training data set did not seem too distorted. The idea behind a single distortion was to seperate out the most discriminative distortion
A single distortion augmentation did not improve perfromance on the validation set.

Next I agumented with images that had multiplicity of distortions per images. See cell #6. This gave ~1% improvement in the validation getting it close to 96% accuracy. Since the training time was large, I did not use augmented data for finding the optimal network parameters
In conclusion, the biggest bang for the buck came from using gray scale images, which bumbed up the accuracy by ~3% to nearly 94%. Creating a balanced data set with augmentation and the extra data with augmentation did not add noticible value.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cell #8 of the ipython notebook. 
My approch was to begin with a smallest network and gradually  increase the size until the validation rate ceased to improve. At this point, additional data or drop out based normalisation should help with overfitting. Unfortunately, and quite inexplicably, while the training error was easily driven to zero, the validation error could not move past 95%. The test error stayed close to validation accuracy. 
Architectures considered were all around LeNet. I tried:

* Larger size of layers
* Addition of dropouts of diffrent values in diffrent networks
* Skip layer architecture 
* Changing the field of vision for various layers

In the end, i decided to stick with plain LeNet model.

I was not expecting the skip layer architecture, mentioned in the refereed paper, to improve the performance as:

1. The network is not deep, so the vanishing gradient is not a issue. Skip layers provide robustness to the vanishing gradient.
3. The images are tigtly cropped, so location within the image is known. This is a detection problem not a segmentation problem.


Dropout did not affect the performance much. I could not train the network for long number of epochs, when the dropout would have mattered. In the short training time, the effect of dropouts did not ick in.
My final model consisted of the a leNet model with  larger convolutinal layers. 



| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution (Conv1)      	|5x5 field, 1x1 stride, No padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution (Conv2)5x5	    | 5x5 field,1x1 stride, No padding, outputs 10x10x64      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		(Fc0)| Input: 1600, output = 120        									|
| Fully connected		(Fc1)| Input: 120, output = 84        									|
| Fully connected		(Fc2)| Input: 84, output = 43        									|
| Softmax				|         									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My approach all along was to avoid hyperparameterise grid sweep. This is because of lack of computational resource, but more importantly, I wanted to develop a systematic,  approach where the I could use the charecterstics of the data, iteratively drive the parameter selection. I performed the following experiments:

1.	Batch size: I tried two batch size, 20 samples and 200 sample. Because the data set is clean (no labeling errors , an assumption), a larger batch size should not have been necessary. I did not find any significant performance impact with batch size. For most experiments I struck to a batch size of 20.
2.	Number of epochs: The accuracy to within 2% of steady state accuracy within the first 10 epochs. So I made quick decisions on feasibility of model by looking at between 20-40 epochs. I did not implement a early stopping rule.
3.	Learning rate: I tried a learning rate of 0.01, which not unexpectedly, resulted in network unable to learn
4.	Drop outs: I tried dropout values of 0.5, 0.75 . I could not break the 95% validation error rate floor, so struck to using no drop outs
In retrospect I have come to realise that this approch is sub-optimal. Deep learning is more experimental and hunting for optimal parameters via experimentation is probably a better approch.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 100%
* validation set accuracy of ~94.5 % 
* test set accuracy of ~93%.

The approximations are due to insufficient cross validation.

##6.If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I used the following steps:

1.	Ensure network learns with just two signs
2.	Scale it up to all hypothesis
3.	Expand network until training error goes to zero. The basic LeNet was sufficient.
4.	Generate an in-data validation set using train-test split.
a.	Check the validation error and change the network size and/or regularization until the validation error improves. It was fairly easy to drive this in-data validation error to < 2%
5.	Check accuracy on the provided validation and train to generalize on this validation set.
a.	This floored at 5% error. This was a mystery that I could not solve. Was the validation data statistically different that training data? Since augmentation did not help, was the augmentation insufficient in generalizing the difference?
b.	Did the validation set have labeling errors?
6.	Check accuracy on test data, but don’t train for the test data
a.	The test data did align well with validation data
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used the following images.

![alt text](/images/459381023.jpg) 
![alt text](/images/459381075.jpg)
![alt text](/images/459381081.jpg)
![alt text](/images/459381091.jpg)
![alt text](/images/images.jpg)
![alt text](/images/459381275.jpg)

2. The images were resized to 32x32 and preprocessed similar to training image
3. On the raw scaled but un-cropped images. The error rate was  1, the network could not identify any signs. 
3. Once I cropped the images into a tight fit around the sign.The network identified the sign with accuracy of 80% 
5. This is explicable as the network is trained only for discrimination and not for localization
6. I accidentally added an image that the network was not trained on. I did not investigate on what the network thought the sign was and why.
7. See notebook, cell # for probabilities associated with images. In general, the network discriminates with an accuracy of ~80% compared to 93% on the test set. It seems the network is not generalised enough.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.


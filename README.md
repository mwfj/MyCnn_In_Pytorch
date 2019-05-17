# Project Description(Midterm Report)

<h3>In my project, I  build a CNN network to do the bulloy action classificaiton.</h3>

## Dataset
In my dataset, I seperate it to two parts. One for test, the other for evaluation. Specifically,  I take 85 percents images in each class for testing(`train_data`), 25 percents  images for evaluating(`val_data`).

In each part, I divded images into 10 categories, they are laughing, pullinghair, quarrel, slapping, punching, stabbing, gossiping, strangle, isolation and nonbullying

**Note that :<br>**

+  the dataset of unbulloying  action comes from [Standford 40 Action](http://vision.stanford.edu/Datasets/40actions.html "Standford 40 Action")
+  I don't have the Copy Right on the bully image datasets, so I don't upload these datasets yet. The Copy Right belongs to [Dr. Feng Luo](https://people.cs.clemson.edu/~luofeng/)

## CNN Model

In my model, I made two covolutional layers and one fully connected layer.
In convolutional layer, I make inchannel in first layer to 3 dimension(Red, Green, Blue) and outchannel to 16. For the second convulotional layer, inchannel  to 16 and outchannel to 32
Both of these convolutional layers are using 3X3 filter size, stride =2 , padding = 1 and 2x2 for max pooling and using ReLU for activation function.
To prevent overfitting, I add BatchNormalization for each convlutional layer and Dropout 50 percents information after ending of each convolutional layer.

In fully connnected layers,  I flated 56*56*32 neurons to do 10 categories classification

## Weight Initialization 
To initilize weight, I use xavier to initial the weight of two convlutional layers,  1 for Batch Normalization layer, 0.01 for fully connected layer.

## Testing & Trainging
I have set learning rate to 0.005, batch_size to 8
In training part, I made epoch number to 35  and print currnt epoch , step and loss after each step
For testing, I have write a test script called 'test.py' and 'mycnn.ckpt' is my model.
Running the command of  `python test.py test.jpg`**(or any other test image)**  to test that script

# The Problem I encounted
## Overfitting
I think my model still have overfiting problem, and my  module's loss function is tending to stable after 20 epoch

## Program problem
When I training module, a warning occasionally throw for me(show below), I took long time to fix it but still not work

```shell
/home/wufangm/.conda/envs/pytorch_env/lib/python3.6/site-packages/PIL/TiffImagePlugin.py:780: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.
  warnings.warn(str(msg))
  
```


<h3>Also, some picture is unreadable during my training time. After I deleted those picture, my module can run nomally, but I am not sure whether it is my program's problem.</h3>



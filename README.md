# -IRISDeep-CNN
![image](https://user-images.githubusercontent.com/87785000/126638350-ce308e14-2b6b-4641-a5bc-3112809b1c87.png)

This Project is done by Eng. Abduallah Damash, and assigned to me by Asst. Prof. Dr. Meryem Erbilek as part of CNG483 (Int. to Comp. Vision) course.

If you have any issues or comments on the project, contact me on Linkedin (https://www.linkedin.com/in/engabduallah).
I also provided the dataset that I used in this project so that you can try it by yourself. 

Insight about the project: 

Implementing an identity recognition system based on convolutional neural networks (CNN) with rectified linear unit (ReLU) as nonlinearity function between 
layers and  use softmax (cross-entropy loss) function to minimize the difference between actual identity and the estimated one. Then, evaluating it with the provided dataset taken from BioSecure Multimodal Database (BMDB) that consists of 200 subjects includes four eye images (two left and two right) for people within the age range of 18-73. Using a high-level language program, particularly Python, and common libraries such as PyTorch OpenCV, Matplotlib, Pandas, and Numpy.

The CNN Architecture: 

![CNN Architecture](https://user-images.githubusercontent.com/87785000/126631134-1d0388e3-cc7c-4236-baee-d7def01d4cda.png)

Hyper Parameters for CNN: 

![image](https://user-images.githubusercontent.com/87785000/126631228-780b3a1c-cd72-4136-b86c-69751ae280ef.png)

Dataset: 
The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilised 
for this project. Four eye images (two left and two right) were acquired in two different sessions with 
a resolution of 640*480 pixels from 200 subjects. Since the left and right eye of an individual is 
completly different. They are considered  as a different individuals. Hence, in this case, database 
will contain 400 subjects each with 4 eye images. Consider the following example for better 
understanding;

![image](https://user-images.githubusercontent.com/87785000/126634787-ad0d73d2-4ad6-41a5-a51a-e8453c81017b.png)

2 samples per person should be used for training set, 1 sample per person should be used for 
validation set and 1 sample per person should be used for testing set. Totaly: 1600 Images. 

Also, the eye images should be cropped to only include 
the iris region by using the given information in parameters.txt. 

You can download the Dataset from the following link: 
https://drive.google.com/file/d/1LToKrSq42xFymFJRzaBrIZuJasTVEgqP/view?usp=sharing

Enjoy it. 

All rights saved, if you want to use any piece of the code, mentioning my name will be enough.

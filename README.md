# Introduction
In this project I've fine tuned the rednet, a semantic segmentator network.
## Project Description


## Goals
The main goals in this project are:
- Teach the newtork to learn a new class images (people)

# Implementation
The network was originally trained on SUNRGBD dataset and the fine-tuned on 100k matterport 3d rgbd samples. From this point, we have further fine tuned the network in order to recognize people. In order to teach people recognition I’ve used a script that trains the RedNet on the total train dataset for ten epochs, using batches of size 32. For each batch, the prediction of the semantic mask was computed and then used to
compute the loss function (cross entropy function). 

The dataset was augmented using different operations such as rotation, horizontal
flip, and vertical flip. All these operations were performed using the library "Albumentations". Albumentations is a free and open-source Python library for image
augmentations
 
# Results

![Alt](/img/rednet_no_people(1).png)
![Alt](/img/rednet_people.png)
Output provided by the not fine-tuned RedNet (top) and the output provided by the fine-tuned RedNet(bottom). As we can notice the not fine-tuned RedNet is not able to recognize correctly the class person by assigning multiple classes to it.

To test how good was the fine-tuned, I measure the general accuracy of the network on all the classe(is the network still able to recognize all the objects?) and the accuracy of the network on only the new class person (has the network learnt the new class?).

## Pre-fine tuning result
- General accuracy: 66,23%
- People accuracy: 2,26%

## Post-fine tuning result
- General accuracy: 63,24%
- People accuracy: 92,2%

These metrics suggest that recognizing a person is generally easier than the average of all the other objects, probably due to the very different shape that a person has compared to the other inanimate objects inside an indoor environment.


# Conclusion
In this project, I've fine tuned the rednet, a semantic segmentator. The goal of this project was to teach to the network to recognize a new class of objects (class person). In this regard, the network was able to reach 92,2% of accuracy for the class person, and 63,24% accuracy in general.

This project was part of a bigger project which goal was to build reinforcement learning agents that were able to navigate inside a digital environment with the presence of people within.
For more information, read the full relation (my master's thesis :): [Socially-aware ObjectGoal Navigation using Proximity-based Auxiliary tasks](https://thesis.unipd.it/retrieve/834673a5-fd91-475f-ba5a-a7327105da67/tesi_franzoso.pdf)

Or consult the [github repository](https://github.com/DavidSolid/SocialObjectNav_main)

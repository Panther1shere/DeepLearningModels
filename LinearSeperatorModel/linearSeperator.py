import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linearSeperationModel import LinearSepearationModel

#creating the dummy data for creating a lineare seperator in order to basically linearly seperate them

noOfSamples=1000
NegativeValues= np.random.multivariate_normal(mean=[0,3],cov=[[1,0.5],[0.5,1]], size=noOfSamples)

PositiveValues= np.random.multivariate_normal(mean=[3,0],cov=[[1,0.5],[0.5,1]], size=noOfSamples)


inputs= np.vstack((NegativeValues, PositiveValues)).astype("float32")
print(f"inputs {inputs} , shape={np.shape(inputs)}")
# generating the equivalent target labels for them

targets=np.vstack((np.zeros((noOfSamples,1)), np.ones((noOfSamples,1)))).astype("float32")

print(targets)


model=LinearSepearationModel(2,1)


model.fit(inputs,targets)

model.calculateAccuracy(inputs, targets)


# getting the models weights and biases for visualising it 
w,b = model.giveModelParameters()


x=np.linspace(-1,4,100)
y= -w[0]/w[1]*x+ (0.5-b)/w[1]
plt.plot(x,y,"-r") # also color that in red basically in order to make it efficient 
plt.scatter(inputs[:,0], inputs[:,1], c= targets[:,0])
plt.savefig("linearSeperationModelWithLine", dpi=300)






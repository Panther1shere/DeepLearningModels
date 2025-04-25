import numpy as np
import tensorflow as tf
import matplotlib as plt

class LinearSepearationModel:
    def __init__(self,inputDim, outputDim, learningrate=0.1):
        self.inputDim=inputDim
        self.outputDim=outputDim
        self.learningrate=learningrate
        self.weights= tf.Variable(tf.random.uniform((inputDim, outputDim)))
        self.bias=tf.Variable(tf.zeros((outputDim,)))

    def __call__(self, inputs):
        pred= self.forwardPass(inputs)
        return pred


    def forwardPass(self, inputs):
        pred= tf.matmul(inputs,self.weights)+self.bias
        return pred


    def fit(self, inputs,labels, epoch=40):
        for i in range(epoch):
            loss= self.trainTheModel(inputs,labels)
            print(f"loss in the epoch {i} is {loss} \n")

        print("finished the training \n ") 

    def meanSquaredError(self, predictions, labels):
        loss= tf.square(labels-predictions)
      
        print(f"shape of loss {tf.shape(tf.reduce_mean(loss))}")
        return tf.reduce_mean(loss)


    def trainTheModel(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions= self.forwardPass(inputs)
            loss=self.meanSquaredError(predictions,labels)

        gradientWRTWeights, gradientWRTBias=tape.gradient(loss,[self.weights, self.bias])
        self.weights.assign_sub(self.learningrate*gradientWRTWeights)
        self.bias.assign_sub(self.learningrate*gradientWRTBias)
        return loss

    def giveModelParameters(self):
        shapeOfWeights= tf.shape(self.weights)
        shapeOfBias=tf.shape(self.bias)
        print(f"shape of both the weights is {shapeOfWeights} and bias is {shapeOfBias}")
        return self.weights, self.bias


    def calculateAccuracy(self, inputs, labels):
        pred = self.forwardPass(inputs)
        pred_classes = tf.where(pred > 0.5, 1.0, 0.0)
        accuracy = tf.reduce_mean(tf.cast(pred_classes == labels, tf.float32))
        print(f"accuracy of the final outputs is {accuracy:.4f}")


    
        

    

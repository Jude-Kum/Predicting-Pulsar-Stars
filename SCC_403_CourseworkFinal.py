##Using supervised learning algorithms to classify pulsar stars.

import numpy as np #importing numpy library for metrics.
import pandas as pd #for datahandling as dataframes
import matplotlib.pyplot as plt #used for data visualization
import scipy as sc
from sklearn.decomposition import PCA

pulsarRaw = []  #creates an empty list of lists to store raw Pulsar data.
pulsarFile = open("HTRU_2.csv", "r") #reads Pulsar text file.

#mushroomFile = open("mushroom.data", "r")
#abaloneFile = open("abalone.dat", "r")

while True:     #a while loop goes through the data line by line.
    theline = pulsarFile.readline()    #reads line 
    if len(theline) == 0:  #checks if the end of the document.
         break     

    readpulsar = theline.split(",")
    for pos in range (len(readpulsar)):
         readpulsar[pos] = float(readpulsar[pos])
    pulsarRaw.append(readpulsar)
    readpulsar = theline.split(",")
    
    #for pos in range (len(readpulsar)):
        #readpulsar[pos] == float(readpulsar[8]
               #classLabels.append(readpulsar)
pulsarFile.close()

pulsar = np.array(pulsarRaw) #converts to a numpy array.

pulsar_C = pulsar[:,:-1] #all columns except class label
pulsar_classLabels = pulsar[:,len(pulsar[0])-1] # keeps class labels only



#pulsar.isnull().any()


#Data Standardisation is carried out before data normalisation. This rescales the data to have a mean of 0 and a 
#standard deviation of 1.

#def standard(pulsar):
    #standardpulsar = pulsar.copy()

    #rows = pulsar.shape[0]
    #cols = pulsar.shape[1]

    #for j in range(cols):
        #sigma = np.std(pulsar[:,j])
        #mu = np.mean(pulsar[: , j])

    #for i in range(rows):
         #standardpulsar[i,j] = (pulsar[i, j] - mu) / sigma

    #eturn standardpulsar



#Data Normalisation to the range [0, 1]. Given the measurement of all 9 feature values are within different ranges,
"""the data is normalised to be able to compare them. This is essential to enable clustering of the data 
based on proximity calculations(for example euclidean distances). The distance metrics used to cluster data
can be distroted if data isnt normalised."""
#Normalising to the [0,1] range.
#def normalise(pulsar):
    #normalisedPulsar = pulsar.copy()
    
    #rows = pulsar.shape[0]
    #cols = pulsar.shape[1]
    
    #for j in range(cols):
        #maxElement = np.amax(pulsar[:,j])
        #minElement = np.amin(pulsar[:, j])
        
    #for i in range(rows):
        #normalisedPulsar[i, j] = (pulsar[i, j]- minElement)/(maxElement - minElement)
 

   #Plotting the Original and standardised data to see how they have been transformed

#def plot(pulsar,fileName):
    #fig,((ax1,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2)

    #fig.set_size_inches(10.0 ,4.0)

    #ax1.plot(pulsar[: ,8],pulsar[: ,0], ".")
    #ax2.plot(pulsar[: ,8],pulsar[: ,1], ".")
    #ax3.plot(pulsar[: ,8],pulsar[: ,2], ".")
    #ax4.plot(pulsar[: ,8],pulsar[: ,3], ".")
    #ax5.plot(pulsar[: ,8],pulsar[: ,4], ".")
    #ax6.plot(pulsar[:, 8],pulsar[:, 5], ".")
    #ax7.plot(pulsar[:, 8],pulsar[:, 6], ".")
    #ax8.plot(pulsar[:, 8],pulsar[:, 7], ".")
    
    #ax1.set_ylabel("Mean of the integrated profile")
    #ax2.set_ylabel("Standard deviation of the integrated profile")
    #ax3.set_ylabel("Excess Kurtosis of the integrated profile")
    #ax4.set_ylabel("Skewness of the integrated profile.")
    #ax5.set_ylabel("Mean of the DM-SNR curve.")
    #ax6.set_ylabel("Standard deviation of the DM-SNR curve.")
    #ax7.set_ylabel("Excess kurtosis of the DM-SNR curve.")
    #ax8.set_ylabel("Skewness of the DM-SNR curve.")

    #ax1.set_xlabel("Class")
    #ax2.set_xlabel("Class")
    #ax3.set_xlabel("Class")
    #ax4.set_xlabel("Class")
    #ax5.set_xlabel("Class")
    #ax6.set_xlabel("Class")
    #ax7.set_xlabel("Class")
    #ax8.set_xlabel("Class")

    #plt.savefig(fileName,bbox_inches='tight')

#plot(pulsar,"notStandardised.pdf")
#plot(standardpulsar,"standardised.pdf") 


def centralize(pulsar_C):
    centralizedPulsar_C = pulsar_C.copy()
    
    rows = pulsar_C.shape[0]
    cols = pulsar_C.shape[1]
    
    for j in range(cols):
        mu = np.mean(pulsar_C[:,j])
        
        for i in range(rows):
            centralizedPulsar_C[i, j] = (pulsar_C[i, j]- mu)
            
    return centralizedPulsar_C

centralizedPulsar_C = centralize(pulsar_C)

#visualizing the correlation between the features
corr_pulsar = np.corrcoef(centralizedPulsar_C, rowvar=0) #rowvar=False ensures each 
                                            #row is an observation 
                                            
#strong positive correlation between the skewness of the integrated
#profile and the excess kurtosis of the integrated profile of the pulsar.
                                            
##PCA converts corr_pulsar to an identity matrix with independent features
                                            
pca_pulsar = PCA(n_components = 5)
pca_pulsar.fit(centralizedPulsar_C)
Coeff_pulsar = pca_pulsar.components_
                                            
transformedPulsar = pca_pulsar.transform(centralizedPulsar_C)   

plt.figure(figsize=(6,4))
plt.plot(transformedPulsar[:,0], transformedPulsar[:,1],".") 

plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

plt.savefig("PCA_pulsar.pdf")



#EXPLAING VARIANCE PERCENTAGE PROVIDED BY EACH PRINCIPAL COMP

plt.figure(figsize=(6,4))

plt.bar([1,2,3,4,5],pca_pulsar.explained_variance_ratio_, tick_label=[1,2,3,4,5])

plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")

PCA_Analysis = plt.savefig("PCAAnalysis.pdf")

###Direct implementation of PCA without using a library
cov_pulsar = np.cov(centralizedPulsar_C)
eigVals, eigVectors = sc.linalg.eig(cov_pulsar)

orderedEigVectors = np.empty(eigVectors.shape)

tmp = eigVals.copy()

maxValue = float("-inf")
maxValuePos = -1

for i in range(len(eigVectors)):

    maxValue = float("-inf")
    maxValuePos = -1
        
    for n in range(len(eigVectors)):
        if (tmp[n] > maxValue):
            maxValue = tmp[n]
            maxValuePos = n

    orderedEigVectors[:,i] = eigVectors[:,maxValuePos]
    tmp[maxValuePos] = float("-inf")

k = 2

#orderedEigVectors[:,1] = -orderedEigVectors[:,1] 
##This inversion will generate the same graph as the one using the library

projectionMatrix = orderedEigVectors[:,0:k]

pcaByHandData = centralizedPulsar_C.dot(projectionMatrix)

plt.figure(figsize=(6,4))

plt.plot(pcaByHandData[:,0],pcaByHandData[:,1],".")

plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

plt.savefig("PCAByHandData.pdf")

plt.close()

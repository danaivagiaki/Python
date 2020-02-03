import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

diatoms = np.loadtxt("diatoms.txt")


def plot_diatom(diatoms, k, col):
    """ Plots diatom shape and interpolates between subsequent landmark points
    
    Args:
        diatoms: a numpy array-like object containing the coordinates of the landmark points of diatom shape
        k: 'int'; the index of the diatoms array; indicates which diatom's shape to plot
        bcol: 'str'; specifies the color of the plot
    """ 

    x = [diatoms[k][i] for i in range(0, len(diatoms[0]), 2)]
    x.append(x[0])
    y = [diatoms[k][j] for j in range(1, len(diatoms[0]), 2)]
    y.append(y[0])
    plt.plot(x, y, color=col, marker="o", mec="darkblue")
    plt.axis("equal")
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")


plt.figure()
plt.title("Shape of the first cell")
plot_diatom(diatoms, 0, "blue")
plt.show()

plt.figure()
for n in range(diatoms.shape[0]):
    plot_diatom(diatoms, n, "blue")
    plt.title("Shapes of all the cells")
plt.show()


# PCA
def pca(data):
    """ Performs principal component analysis (PCA) on input data. Returns a numpy array of the PCs and a numpy array of the variance of each PC in decreasing order

    Args:                                                                             
        data: a numpy array-like object                                                            
    """
    # We are interested in the variance between the different features, i.e. the columns 
    Sigma = np.cov(data, rowvar=False)
    evals, evecs = LA.eig(Sigma)
    # Make sure the variance is monotonically decreasing:                                    
    dc_dict = {}
    for l in range(evals.shape[0]):
        dc_dict[evals[l]] = evecs[:,l]
    evals = sorted(evals)
    evals_decr = np.flip(evals, axis=0)
    evecs_decr = np.array([dc_dict[val_decr] for val_decr in evals_decr])
    evecs_decr = evecs_decr.T

    return evecs_decr, evals_decr


eigvecs = pca(diatoms)[0]
eigvals = pca(diatoms)[1]
sd = np.sqrt(eigvals)
means = np.array([np.mean(diatoms[:, col]) for col in range(diatoms.shape[1])])
blues = plt.get_cmap("Blues")

pc1_1 = np.subtract(means, 2*sd[0]*eigvecs[:,0])
pc1_2 = np.subtract(means, sd[0]*eigvecs[:,0])
pc1_4 = np.add(means, sd[0]*eigvecs[:,0])
pc1_5 = np.add(means, 2*sd[0]*eigvecs[:,0])
pc1 = [pc1_1, pc1_2, means, pc1_4, pc1_5]

pc2_1 = np.subtract(means, 2*sd[1]*eigvecs[:,1])
pc2_2 = np.subtract(means, sd[1]*eigvecs[:,1])
pc2_4 = np.add(means, sd[1]*eigvecs[:,1])
pc2_5 = np.add(means, 2*sd[1]*eigvecs[:,1])
pc2 = [pc2_1, pc2_2, means, pc2_4, pc2_5]

pc3_1 = np.subtract(means, 2*sd[2]*eigvecs[:,2])
pc3_2 = np.subtract(means, sd[2]*eigvecs[:,2])
pc3_4 = np.add(means, sd[2]*eigvecs[:,2])
pc3_5 = np.add(means, 2*sd[2]*eigvecs[:,2])
pc3 = [pc3_1, pc3_2, means, pc3_4, pc3_5]


def plot_PC(pc_list, pc):
    """ Plots instances of some principal component.
    
    Args: 
        pc_list: a 'list' containing the instances to plot
        pc: 'str'; name of the principal component, e.g. "PC1"
    """

    plt.figure()
    c = 0.2
    for l in range(len(pc_list)):
        plot_diatom(pc_list, l, blues(c))
        c+=0.2      
        plt.title("Instances of %s" % pc)
    plt.show()


plot_PC(pc1, "PC1")
plot_PC(pc2, "PC2")
plot_PC(pc3, "PC3")


# Plotting cumvar vs. PC
training_data = np.loadtxt("./IDSWeedCropTrain.csv", delimiter=",")
Xtrain = training_data[:, :-1]
eigvecs3 = pca(Xtrain)[0]
eigvals3 = pca(Xtrain)[1]
normalised_data = (Xtrain - Xtrain.mean(axis=0))/Xtrain.std(axis=0)
Sig2 = np.dot(normalised_data.T, normalised_data)/normalised_data.shape[0]
evas, eves = LA.eig(Sig2)
dc = {}
for l in range(evas.shape[0]):
    dc[evas[l]] = eves[:,l]
evas = sorted(evas)
eigvals31 = np.flip(evas, axis=0)
evecs_decr = np.array([dc[val_decr] for val_decr in eigvals31])
eigvecs31 = evecs_decr.T

plt.figure()
plt.subplot(2,1,1)
cum_var = np.cumsum(eigvals3/np.sum(eigvals3))
plt.plot(np.arange(1, eigvecs3.shape[0]+1, 1), cum_var, marker="o", mfc="blue", mec="darkblue")
plt.xticks(np.arange(1, eigvecs3.shape[0]+1, 1))
plt.title("Cumulative normalised variance vs. PC number on non-standardised data")
plt.xlabel("Principal component number")
plt.ylabel("Cumulative variance")
plt.subplot(2,1,2)
cum_var2 = np.cumsum(eigvals31/np.sum(eigvals31))
plt.plot(np.arange(1, eigvecs31.shape[0]+1, 1), cum_var2, marker="o", mfc="blue", mec="darkblue")
plt.xticks(np.arange(1, eigvecs31.shape[0]+1, 1))
plt.title("Cumulative normalised variance vs. PC number on standardised data")
plt.xlabel("Principal component number")
plt.ylabel("Cumulative variance")
plt.tight_layout()
plt.show()


## KMeans Clustering
# Divide the pesticide training data into two classes
Xclass = training_data[:, -1]
class1 = Xtrain[np.nonzero(Xclass)]
class0 = Xtrain[np.where(Xclass == 0)]


def k_means(data, k):
    """ Clusters the input data into k subsets. Returns the k subsets and the mean of each subset
    
    Args:
        data: a numpy array-like object, where columns are the variables and rows are the different observations
        k: 'int'; the number of centroids (means) to be used for the clustering
    """
    # Initialise k centroids
    for m in range(k):
        means = [data[m,:] for m in range(k)]
        datasets = [data[m,:] for m in range(k)]
    # Initialise clusters
    # Find the centroid that minimizes ||μ-x||
    for i in range(k, data.shape[0]):
        distances = [LA.norm(np.subtract(data[i,:], m)) for m in means]
        m_idx = np.argmin(np.array(distances))
        datasets[m_idx] = np.vstack((datasets[m_idx], data[i,:]))
    # Calculate new centroids (cluster means)
    # Initialise each new centroid as the mean of the first column of each cluster 
    means_new = [np.mean(datasets[p][:,0]) for p in range(len(datasets))]
    # Each cluster mean is a row vector containing the mean of every column of the cluster datapoints
    for s in range(len(datasets)):
        mnew = means_new[s]
        for column in range(1, datasets[s].shape[1]):
            colm = np.mean(datasets[s][:, column])
            mnew = np.hstack((mnew, colm))
        means_new[s] = mnew
    # Iterate until the centroids do not change; for non-changing given data, same centroids is equivalent to same clusters
    iteration = 0
    while np.any(np.array([np.any(means_new[m] != means[m]) for m in range(len(means))])): 
        iteration += 1
       # print(iteration)
        means = means_new
        datasets = [None]*k
        # Data assigment
        for j in range(data.shape[0]):
            distances = [LA.norm(np.subtract(data[j,:], m)) for m in means]
            m_idx = np.argmin(np.array(distances))
            if np.any(datasets[m_idx] == None):                                       
                datasets[m_idx] = data[j,:]                                  
            else:
                datasets[m_idx] = np.vstack((datasets[m_idx], data[j,:])) 
        means_new = [np.mean(datasets[p][:,0]) for p in range(len(datasets))]
        # Centroid relocation
        for s in range(len(datasets)):
            mnew = means_new[s]
            for c in range(1, datasets[s].shape[1]):
                colm = np.mean(datasets[s][:, c])
                mnew = np.hstack((mnew, colm))
            means_new[s] = mnew

    return means_new 


centers  = k_means(Xtrain, 2)
print("The center of my first cluster is:")
print(centers[0])
print("The center of my second cluster is:")
print(centers[1])

# Project the training data onto the first two PCs
proj1_class1 = np.dot(class1, eigvecs3[:,0])
proj1_class0 = np.dot(class0, eigvecs3[:,0])
proj2_class1 = np.dot(class1, eigvecs3[:,1])
proj2_class0 = np.dot(class0, eigvecs3[:,1])
# Project the two cluster centres onto the first two PCs
proj1_center1 = np.dot(centers[0], eigvecs3[:,0])
proj1_center2 = np.dot(centers[1], eigvecs3[:,0])
proj2_center1 = np.dot(centers[0], eigvecs3[:,1])
proj2_center2 = np.dot(centers[1], eigvecs3[:,1])
proj_center1 = np.vstack((proj1_center1, proj2_center1))
proj_center2 = np.vstack((proj1_center2, proj2_center2))
print("The projection of the first cluster centre is:")
print(proj_center1) 
print("The projection of the second cluster centre is:")
print(proj_center2) 

# Plot the projections of the training data and of the cluster centres
plt.figure()
plt.scatter(proj1_class1, proj2_class1, color="red", label="class 1")
plt.scatter(proj1_class0, proj2_class0, color="green", label="class 0")
plt.scatter(proj1_center1, proj2_center1, color="darkblue", s=80, label="1st cluster center")
plt.scatter(proj1_center2, proj2_center2, color="black", s=80, label="2nd cluster center")
plt.title("Projection of pesticide data onto the first 2 PCs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

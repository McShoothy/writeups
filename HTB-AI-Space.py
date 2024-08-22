import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Load the distance matrix from the file 'distance_matrix.npy'
distance_matrix = np.load('distance_matrix.npy')

# Initialize the MDS model
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)

mds_results = mds.fit_transform(distance_matrix)

# Create a scatter plot to visualize the MDS results
plt.figure(figsize=(8, 6))
plt.scatter(mds_results[:, 0], mds_results[:, 1], s=100)

plt.title('Multidimensional Scaling (MDS) Results')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid(True)

# Display the plot
plt.show()

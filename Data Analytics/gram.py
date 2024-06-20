import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the documents
documents = [
    "I love playing football.",
    "Football is a great sport.",
    "I love watching movies.",
    "The weather today is sunny.",
    "Reading books is my hobby.",
    "I enjoy traveling to new places.",
    "Coffee is my favorite drink.",
    "She loves to bake cookies.",
    "They are planning a vacation.",
    "Learning new languages is fun.",
    "Cooking is an essential skill.",
    "Exercise is important for health.",
    "He is a professional dancer.",
    "Gardening is very relaxing.",
    "Jogging in the park is refreshing.",
    "She enjoys painting landscapes.",
    "He is a talented musician.",
    "They are organizing a concert.",
    "Writing poetry is his passion.",
    "The movie was thrilling.",
    "She is an avid reader.",
    "Technology is advancing rapidly.",
    "Nature walks are rejuvenating.",
    "He loves solving puzzles.",
    "Cycling is a great workout.",
    "Swimming is a complete exercise.",
    "He is an excellent chef.",
    "Photography captures moments.",
    "She is learning to play guitar.",
    "Yoga helps in maintaining balance.",
    "The cat is sleeping peacefully.",
    "Traveling broadens the mind.",
    "They are launching a startup.",
    "Meditation calms the mind.",
    "She enjoys crafting handmade items.",
    "The dog is very playful.",
    "Camping under the stars is amazing.",
    "He is studying computer science.",
    "She loves to collect stamps.",
    "The book was very insightful.",
    "Running marathons is challenging.",
    "She is a fitness enthusiast.",
    "He enjoys playing chess.",
    "She is learning martial arts.",
    "Bird watching is a relaxing activity.",
    "Hiking in the mountains is exhilarating.",
    "He enjoys building model airplanes.",
    "She is a volunteer at the animal shelter.",
    "The museum has fascinating exhibits.",
    "He is passionate about photography."
]

# Initialize the CountVectorizer with lowercase conversion and removal of punctuation
vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')

# Tokenize and build the term frequency matrix
X = vectorizer.fit_transform(documents)

# Convert to array to see the term frequencies
term_frequency_matrix = X.toarray()

# Get the terms
terms = vectorizer.get_feature_names_out()

# Create a term frequency DataFrame
df = pd.DataFrame(term_frequency_matrix, columns=terms)

# Normalize the term frequencies (TF) by the total number of terms in each document
df = df.div(df.sum(axis=1), axis=0)

# Transpose the DataFrame for easier reading
df = df.T
df.columns = [f"Document #{i+1} TF" for i in range(len(documents))]

# Sort the DataFrame by term names
df = df.sort_index()

print("Term Frequency Table:")
print(df)

# Number of documents
num_documents = len(documents)

# Count the number of documents containing each term
doc_freq = (X > 0).sum(axis=0).A1

# Compute IDF values
idf_values = np.log(num_documents / (1 + doc_freq))

# Create a DataFrame for IDF values
idf_df = pd.DataFrame(idf_values, index=terms, columns=["IDF"])

# Sort the DataFrame by term names
idf_df = idf_df.sort_index()

print("IDF Table:")
print(idf_df)

# Compute TF-IDF
tf_idf_matrix = df.mul(idf_df["IDF"], axis=0)

print("TF-IDF Term-Document Matrix:")
print(tf_idf_matrix)

# Ensure the TF-IDF matrix is correctly shaped
tf_idf_matrix_np = tf_idf_matrix.to_numpy()

# Compute the Gram matrix (X^T * X)
gram_matrix = np.dot(tf_idf_matrix_np.T, tf_idf_matrix_np)

print("Gram Matrix (X^T * X):")
print(gram_matrix)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Normalize the eigenvectors (each column)
eigenvectors_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Transpose to get V^T
V_T = eigenvectors_normalized.T

print("Eigenvalues:")
print(eigenvalues)
print("Normalized Eigenvectors (V^T):")
print(V_T)

# Compute the singular values (square roots of the eigenvalues)
singular_values = np.sqrt(eigenvalues)

# Create the diagonal matrix Σ
sigma_matrix = np.diag(singular_values)

print("Singular Values:")
print(singular_values)
print("Sigma Matrix (Diagonal Matrix of Singular Values):")
print(sigma_matrix)

# Convert Σ to DataFrame for better readability
sigma_df = pd.DataFrame(sigma_matrix)

print("Sigma Matrix DataFrame:")
print(sigma_df.to_string())

# Compute the inverse of Sigma matrix
sigma_inv_matrix = np.linalg.inv(sigma_matrix)

# Compute U as U = X V Sigma^-1
U = np.dot(tf_idf_matrix_np, np.dot(V_T.T, sigma_inv_matrix))

print("Matrix U:")
print(U)

# Convert U to DataFrame for better readability
U_df = pd.DataFrame(U, columns=[f"U{i+1}" for i in range(U.shape[1])])

print("U DataFrame:")
print(U_df.to_string())

# Reconstruct X using U, Sigma, and V^T
reconstructed_X = np.dot(U, np.dot(sigma_matrix, V_T))

print("Reconstructed X:")
print(reconstructed_X)

# Convert original X to a DataFrame for better comparison
original_X_df = pd.DataFrame(tf_idf_matrix_np, columns=[f"Term {i+1}" for i in range(tf_idf_matrix_np.shape[1])])

print("Original X DataFrame:")
print(original_X_df.to_string())

# Convert reconstructed X to a DataFrame for better comparison
reconstructed_X_df = pd.DataFrame(reconstructed_X, columns=[f"Term {i+1}" for i in range(reconstructed_X.shape[1])])

print("Reconstructed X DataFrame:")
print(reconstructed_X_df.to_string())

# Check if the reconstructed X is approximately equal to the original X
if np.allclose(reconstructed_X, tf_idf_matrix_np):
    print("The reconstruction is successful and X is approximately equal to U * Sigma * V^T.")
else:
    print("The reconstruction is not successful. There are differences between X and U * Sigma * V^T.")

# Dimensionality reduction: retaining the top k singular values and corresponding vectors
k = 2  # Number of components to retain

# Select the top k components
U_k = U[:, :k]
Sigma_k = sigma_matrix[:k, :k]
V_T_k = V_T[:k, :]

# Reconstruct the reduced matrix A_k
A_k = np.dot(U_k, np.dot(Sigma_k, V_T_k))

print(f"Reconstructed Matrix A_k with top {k} components:")
print(A_k)

# Convert A_k to DataFrame for better readability
A_k_df = pd.DataFrame(A_k, columns=[f"Term {i+1}" for i in range(A_k.shape[1])])

print(f"Reconstructed A_k DataFrame with top {k} components:")
print(A_k_df.to_string())

# Plot heatmap of the original TF-IDF matrix
plt.figure(figsize=(12, 8))
sns.heatmap(tf_idf_matrix_np, annot=True, cmap='viridis', xticklabels=[f"Document #{i+1}" for i in range(len(documents))], yticklabels=terms)
plt.title('Original TF-IDF Matrix')
plt.show()

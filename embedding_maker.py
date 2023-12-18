import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Read the CSV file
df = pd.read_csv('test.csv')

# Generate embeddings
context_embeddings = model.encode(df['context'].tolist(), convert_to_tensor=True)
ans_embeddings = model.encode(df['ans'].tolist(), convert_to_tensor=True)

# Convert embeddings to lists (to store in DataFrame)
context_embeddings_list = [embedding.numpy() for embedding in context_embeddings]
ans_embeddings_list = [embedding.numpy() for embedding in ans_embeddings]

# Create a new DataFrame
new_df = pd.DataFrame({
    'label': df['label'],
    'context_embedding': context_embeddings_list,
    'ans_embedding': ans_embeddings_list
})

# Save to a new CSV file
new_df.to_csv('output_embeddings.csv', index=False)

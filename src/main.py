import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from unpack import convert_xl_to_df

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File and sheet name for loading data
file_name = "data/protein_solubility_esol.xlsx"
sheet_name = "Sheet1"

# Load the data from the Excel file
logging.info("Loading data from Excel file...")
df = convert_xl_to_df(file_name, sheet_name)
sequences = df.get("sequence").to_list()
solubility = df.get("solubility").to_list()

logging.info("Data loaded successfully.")
logging.info(f"Number of sequences: {len(sequences)}")

# Function to generate k-mers from a protein sequence
def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Convert each protein sequence into a list of k-mers and join them with spaces
k = 3
logging.info(f"Generating k-mers with k={k}...")
sequences_kmers = [" ".join(get_kmers(seq, k)) for seq in sequences]
logging.info("k-mers generated successfully.")

# Apply Count Vectorization to the k-mer representations
logging.info("Applying Count Vectorization...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sequences_kmers)
logging.info("Count Vectorization applied successfully.")

# Split the data into training and testing sets
logging.info("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, solubility, test_size=0.1, random_state=42)
logging.info(f"Training set size: {len(y_train)}, Testing set size: {len(y_test)}")

# Train the Support Vector Regressor
logging.info("Training the Support Vector Regressor...")
svr = SVR(kernel='linear')  # You can try different kernels like 'rbf' for experimentation
svr.fit(X_train, y_train)
logging.info("Model trained successfully.")

# Predict solubility values on the test set
logging.info("Making predictions on the test set...")
y_pred = svr.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
logging.info(f"Mean Squared Error: {mse:.4f}")

# Display results
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
logging.info("Results generated successfully. Displaying first few rows of the results:")
print(results.head())
# CSCI 555: The Toxicity Prediction Challenge II
# Name: Sanket Dilip Vagal
# StFX Student ID Number: 202207184
# StFX email ID: x2022fjg@stfx.ca

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, MolSurf, PandasTools
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data into a pandas DataFrame
train = pd.read_csv(Path(__file__).resolve().parent / "input/train_II.csv")
test = pd.read_csv(Path(__file__).resolve().parent / "input/test_II.csv")

# Split the 'Id' column into 'SMILES' and 'Assay ID' columns for both train and test dataframes
train[["SMILES", "Assay ID"]] = train["Id"].str.split(";", expand=True)
train = train.drop("Id", axis=1)

test = test["x"].str.split(";", expand=True)
test.columns = ["SMILES", "Assay ID"]

# Convert 'Assay ID' column to numeric type for both train and test dataframes
train["Assay ID"] = pd.to_numeric(train["Assay ID"])
test["Assay ID"] = pd.to_numeric(test["Assay ID"])

# Convert 'SMILES' column to RDKit mol object for both train and test dataframes
PandasTools.AddMoleculeColumnToFrame(train, smilesCol="SMILES")
PandasTools.AddMoleculeColumnToFrame(test, smilesCol="SMILES")

# Drop rows with null values in the 'ROMol' column for both train and test dataframes
train = train[train["ROMol"].notnull()]
test = test[test["ROMol"].notnull()]

# Store the 'Expected' column in 'labels' variable and drop the 'Expected' column from the train dataframe
labels = train["Expected"]
train = train.drop("Expected", axis=1)

# Define a function to calculate new features from SMILES using RDKit
def calculate_features(smiles: pd.Series) -> pd.Series:
    mol = Chem.MolFromSmiles(smiles)
    features = {}
    features["num_atoms"] = mol.GetNumAtoms()
    features["mol_weight"] = Descriptors.MolWt(mol)
    features["logp_i"] = Descriptors.MolLogP(mol)
    features["h_bond_donor_i"] = Descriptors.NumHDonors(mol)
    features["h_bond_acceptors_i"] = Descriptors.NumHAcceptors(mol)
    features["rotb"] = Descriptors.NumRotatableBonds(mol)
    features["tpsa"] = Descriptors.TPSA(mol)
    features["heavy"] = mol.GetNumHeavyAtoms()
    radius = 2  # set the radius of the Morgan fingerprint
    nbits = 1024  # set the number of bits in the Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    morgan = fp.ToBitString()
    for i, bit in enumerate(morgan):
        features[f"morgan_{i}"] = int(bit)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    for i, bit in enumerate(maccs):
        features[f"maccs_{i}"] = bit
    features["aromatic_cc"] = Lipinski.NumAromaticCarbocycles(mol)
    features["peoe_vsa1"] = MolSurf.PEOE_VSA1(mol)
    return pd.Series(features)


# Apply the function to the SMILES column to create new features for train data
new_features = train["SMILES"].apply(calculate_features)

# Generate column names based on feature values
feature_names = [f"{feat}" for feat in new_features.columns]

# Assign the new features to the DataFrame with the generated column names
train[feature_names] = new_features

# Apply the function to the SMILES column to create new features for test data
new_features = test["SMILES"].apply(calculate_features)

# Generate column names based on feature values
feature_names = [f"{feat}" for feat in new_features.columns]

# Assign the new features to the DataFrame with the generated column names
test[feature_names] = new_features

# Drop the "SMILES" and "ROMol" columns from the train and test datasets
train = train.drop("SMILES", axis=1)
test = test.drop("SMILES", axis=1)
train = train.drop("ROMol", axis=1)
test = test.drop("ROMol", axis=1)

# Create a VarianceThreshold object to remove features with low variance
var_threshold = VarianceThreshold(threshold=0.2)

# Fit the VarianceThreshold object to the training data
var_threshold.fit(train)

# Get the indices of the important features
important_features = train.columns[var_threshold.get_support()]
print(len(important_features), important_features)

# Select only the important features from the train and test datasets
train_s = train.loc[:, important_features]
test_s = test.loc[:, important_features]

# Create a LightGBM classifier with best hyperparameters obtained using GridSearchCV
lgb_clf_1 = lgb.LGBMClassifier(
    max_depth=12, n_estimators=1400, scale_pos_weight=0.6, random_state=42
)

# Use StratifiedKFold cross-validation for model evaluation with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train the model using cross-validation and print the mean F1-macro score
scores = cross_val_score(
    lgb_clf_1, train_s, labels, cv=skf, scoring="f1_macro", verbose=1, n_jobs=-1
)

print("Mean F1 Score: ", sum(scores) / len(scores))

# Fit the model on the entire training set
lgb_clf_1.fit(train_s, labels)

# Make predictions on the test data using the trained LGBM classifier
predictions = lgb_clf_1.predict(test_s)

# Read the test data from a CSV file and create a DataFrame for the predictions
test_data = pd.read_csv(Path(__file__).resolve().parent / "input/test_II.csv")
output = pd.DataFrame({"Id": test_data.x, "Predicted": predictions})

# Save the predictions to a CSV file
output.to_csv(Path(__file__).resolve().parent / "submission.csv", index=False)
print("Your submission was successfully saved!")

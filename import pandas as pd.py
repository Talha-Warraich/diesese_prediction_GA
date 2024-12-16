import pandas as pd

# Load the diseaseandsymptom file
disease_symptom_file = pd.read_csv('DiseaseAndSymptoms.csv')  # Replace with your actual file path

# Step 1: Identify unique symptoms
symptom_columns = [col for col in disease_symptom_file.columns if 'Symptom' in col]
unique_symptoms = set()


for symptom_col in symptom_columns:
    unique_symptoms.update(disease_symptom_file[symptom_col].dropna().unique())
print(unique_symptoms)
# Step 2: Create a new DataFrame with diseases and unique symptoms
updated_data = pd.DataFrame(columns=['Disease'] + list(unique_symptoms))

# Step 3: Populate the new DataFrame
rows = []  # List to hold the new rows
for index, row in disease_symptom_file.iterrows():
    disease = row['Disease']
    new_row = {'Disease': disease}
    
    for symptom in unique_symptoms:
        new_row[symptom] = 1 if symptom in row.values else 0  # Mark presence with 1, absence with 0
    
    rows.append(new_row)  # Append the new row to the list

# Convert the list of rows to a DataFrame
updated_data = pd.DataFrame(rows)

# Step 4: Save the updated data to a new Excel file
updated_file_path = 'updated_diseaseandsymptom.xlsx'
updated_data.to_excel(updated_file_path, index=False)

print(f"Updated file '{updated_file_path}' created successfully.")
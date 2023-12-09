import warnings


def custom_warning_filter(message, category, filename, lineno, file=None, line=None):
    return "DecisionTreeClassifier" not in str(message)


warnings.showwarning = custom_warning_filter

import pandas as pd
import numpy as np

df = pd.read_csv("D:\HDHI Admission data.csv")
merged_wavg = pd.read_csv("D:\Drug_Recommender.csv")

df['HB'].replace('EMPTY', np.nan, inplace=True)
df['TLC'].replace('EMPTY', np.nan, inplace=True)
df['PLATELETS'].replace('EMPTY', np.nan, inplace=True)
df['UREA'].replace('EMPTY', np.nan, inplace=True)
df['CREATININE'].replace('EMPTY', np.nan, inplace=True)
df['GLUCOSE'].replace('EMPTY', np.nan, inplace=True)

null_pct = df.apply(pd.isnull).sum()/df.shape[0]

valid_columns = df.columns[null_pct < .06]

df =  df[valid_columns].copy()
df =  df.ffill()
df.apply(pd.isnull).sum()

indep = df.drop(['SNO', 'MRD No.', 'D.O.A', 'D.O.D', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'month year', 'DURATION OF STAY', 'duration of intensive unit stay', 'OUTCOME', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT', 'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC', 'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK', 'PULMONARY EMBOLISM', 'CHEST INFECTION'],axis='columns')
dept = df[['SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT', 'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC', 'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK', 'PULMONARY EMBOLISM', 'CHEST INFECTION']]

from sklearn.preprocessing import LabelEncoder
la_GENDER=LabelEncoder()
indep['GENDER_n']=la_GENDER.fit_transform(indep['GENDER'])

Indep_n = indep.drop('GENDER',axis='columns')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

X = Indep_n[['AGE', 'SMOKING ', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'RAISED CARDIAC ENZYMES', 'GENDER_n']]
y = dept[['SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT', 'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC', 'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK', 'PULMONARY EMBOLISM', 'CHEST INFECTION']]  # Replace with your actual target columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import GridSearchCV

classifier = DecisionTreeClassifier(random_state=42)


param_grid = {
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}


grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_


best_classifier = DecisionTreeClassifier(random_state=42, **best_params)
best_classifier.fit(X_train, y_train)


accuracy = best_classifier.score(X_test, y_test)


classifier = DecisionTreeClassifier()


model = MultiOutputClassifier(classifier)


model.fit(X_train, y_train)


predictions = model.predict(X_test)


print("Give the inputs as per the example, use only LOWERCASE")
age_in = int(input("Enter your age (eg: 19,56): "))
smoker_in = input("Mention whether you are smoker are not (eg: no,yes): " )
alcohol_in = input("Mention whether you are alcoholic or not (eg: no,yes): ")
diabetes_in = input("Mention whether you have diabetes or not (eg: no,yes): ")
hypertension_n = input("Mention whether you have hypertension or not (eg: no,yes): ")
artery_in = input("Mention whether you have coronary artery disease or not (eg: no,yes): ")
cardiac_in = input("Mention whether you have prior cardiac events or not (eg: no,yes): ")
kidney_in = input("Mention whether you have kidney disease or not (eg: no,yes): ")
hemoglobin_in = float(input("Mention your hemoglobin count (eg: 13.7): "))
leukocyte_in = float(input("Mention your leukocyte count (eg: 14.7): "))
platelete_count_in = float(input("Mention your platelets count(eg: 334): "))
glucose_in = float(input("Mention your glucose count (eg: 144): "))
urea_in = float(input("Mention your urea count (eg: 55): "))
creatinine_in = float(input("Mention your creatinine count (eg: 1.4): "))
cardiac_enzy_in = input("Mention whether you have raised cardiac enzymes or not (eg: no,yes): ")
gender_in = input("Mention your gender (eg: male,female): ")

smoker_prob_list = [smoker_in]
alcohol_prob_list = [alcohol_in]
diabetes_prob_list = [diabetes_in]
hypertension_prob_list = [hypertension_n]
artery_prob_list = [artery_in]
cardiac_prob_list = [cardiac_in]
kidney_prob_list = [kidney_in]
cardiac_enzy_list = [cardiac_enzy_in]
gender_list = [gender_in]

smoke_p = 0
alcohol_p = 0
diabetes_p = 0
hypertension_p = 0
artery_p = 0
cardiac_p = 0
kidney_p = 0
cardiac_enzy_p = 0
gender_p = 0

for smoke_d in smoker_prob_list:
    if smoke_d == 'yes':
        smoke_p = 1
    else:
        smoke_p = 0
        

for alcohol_d in alcohol_prob_list:
    if alcohol_d =='yes':
        alcohol_p = 1
    else:
        smoke_p = 0
        
for diabetes_d in diabetes_prob_list:
    if diabetes_d =='yes':
        diabetes_p = 1
    else:
        diabetes_p = 0
        
for hypertension_d  in hypertension_prob_list:
    if hypertension_d =='yes':
        hypertension_p = 1
    else:
        hypertension_p = 0
        
for artery_d in artery_prob_list:
    if artery_d == 'yes':
        artery_p = 1
    else:
        artery_p = 0
        
for cardiac_d in cardiac_prob_list:
    if cardiac_d == 'yes':
        cardiac_p = 1
    else:
        cardiac_p = 0
        
for kidney_d in kidney_prob_list:
    if kidney_d == 'yes':
        kidney_p = 1
    else:
        kidney_p = 0
        
for cardiac_enzy_d in cardiac_enzy_list:
    if cardiac_enzy_d == "yes":
        cardiac_enzy_p = 1
    else:
        cardiac_enzy_p = 0
        
for gender_d in gender_list:
    if gender_d == "male":
        gender_p = 1
    else:
        gender_p = 0
    

prediction = model.predict([[age_in,smoke_p,alcohol_p,diabetes_p,hypertension_p,artery_p,cardiac_p,kidney_p,hemoglobin_in,leukocyte_in,platelete_count_in,glucose_in,urea_in,creatinine_in,cardiac_enzy_p,gender_p]])
conditions = [
    "Severe_Anemia", "Anaemia", "Stable_Angina", "Acute_Coronary_Syndrome", "STEMI",
    "Atypical_chest_pain", "Heart_Failure", "Heart_Failure_Reduced_EF", "Heart_Failure_preserved_EF",
    "Valvular_Heart_Disease", "Complete_Heart_Block", "Sick_Sinus_Syndrome", "Acute_Kidney_Injury",
    "Ischemic_Stroke", "Hemorrhagic_Stroke", "Atrial_Fibrillation","Ventricular_Tachycardia",
    "PSVT", "Congential_heart_Disease", "Urinary_Tract","Neuro_Cardiogenic_Syncope",
    "Orthostatic","Infective_Endocarditis","DVT","Cardiogenic_Shock",
    "Shock", "Pulmonary_Embolism","Chest_Infection"]
 
predicted_diseases = []
print(" ")
print(" ")
print("----------------------------------------------------------")
for i in range(len(prediction[0])):
    if prediction[0][i] == 1:
        print(f"You may be affected with {conditions[i]}.")
        predicted_disease = conditions[i]
        predicted_diseases.append(predicted_disease)
print("----------------------------------------------------------")
print(" ")        
condition_mapping = {
    "Severe_Anemia": "Anemia",
    "Anaemia": "Anemia",
    "Stable_Angina": "Angina",
    "Acute_Coronary_Syndrome": "Acute Coronary Syndrome",
    "STEMI": "Heart Attack",
    "Atypical_chest_pain": "Gastroenteritis",
    "Heart_Failure": "Heart Failure",
    "Heart_Failure_Reduced_EF": "Heart Failure",
    "Heart_Failure_preserved_EF": "Heart Failure",
    "Valvular_Heart_Disease": "Prosthetic Heart Valves, Mechanical Valves - Thrombosis Prophylaxis",
    "Complete_Heart_Block": "Heart Attack",
    "Sick_Sinus_Syndrome": "Sinusitis",
    "Acute_Kidney_Injury": "Kidney Infections",
    "Ischemic_Stroke": "Ischemic Stroke",
    "Hemorrhagic_Stroke": "Influenza Prophylaxis",
    "Atrial_Fibrillation": "Atrial Fibrillation",
    "Ventricular_Tachycardia": "Supraventricular Tachycardia",
    "PSVT": "Supraventricular Tachycardia",
    "Congential_heart_Disease": "Heart Attack",
    "Urinary_Tract": "Urinary Tract Infection",
    "Neuro_Cardiogenic_Syncope": "Neuropathic Pain",
    "Orthostatic": "Postural Orthostatic Tachycardia Syndrome",
    "Infective_Endocarditis": "Bacterial Endocarditis Prevention",
    "DVT": "Deep Vein Thrombosis",
    "Cardiogenic_Shock": "Heart Attack",
    "Shock": "Brain Tum",
    "Pulmonary_Embolism": "Pulmonary Embolism",
    "Chest_Infection": "Bronchitis"
}


from collections import defaultdict

predicted_diseases_recom = [condition_mapping[condition] for condition in predicted_diseases]


condition_groups = defaultdict(list)
for condition, original_condition in zip(predicted_diseases_recom, predicted_diseases):
    condition_groups[condition].append(original_condition)

merged_wavg = merged_wavg.drop(merged_wavg[merged_wavg['Review_Sentiment'] == "Negative"].index)
merged_wavg = merged_wavg.drop(merged_wavg[merged_wavg['Review_Sentiment'] == "Neutral"].index)

groupedByCount = merged_wavg.groupby(['condition', 'drugName', 'Rating_Wavg'])['usefulCount'].sum().reset_index()


for condition, original_conditions in condition_groups.items():
    groupedByDisease = groupedByCount.groupby('condition')
    if condition in groupedByDisease.groups:
        group_data = groupedByDisease.get_group(condition)
        predicted_drug = pd.DataFrame(group_data.nlargest(3, ['Rating_Wavg', 'usefulCount']))
        print(f"Recommended drugs for {', '.join(original_conditions)}:")
        print(" ")
        for index, row in predicted_drug.iterrows():
            drug_name = row['drugName']
            useful_count = row['usefulCount']
            print(f" {drug_name}")
            print(f"  {useful_count} - found it usefull")
            print("  ")
    else:
        print(f"No data available for {', '.join(original_conditions)}")
    print("------------------------")  


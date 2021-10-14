# Import libraries
from scipy.io import arff
import pandas as pd
import numpy as np
import pandasgui
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

# Load data from file
data, meta = arff.loadarff('dataset/Autism-Child-Data.arff')
df = pd.DataFrame(data)

# Remove b'..' from columns
str_df = df.select_dtypes([object])
str_df = str_df.stack().str.decode('utf-8').unstack()

# Display raw data in pandasgui
# pandasgui.show(str_df, settings={'block': True})

# Create labels - autism yes/no
labels = np.where(df[['austim']].values == 'yes', 1, 0)
labels = labels.reshape(292)

# Drop unused columns
df = str_df.drop(columns=['age_desc', 'austim', 'contry_of_res', 'ethnicity', 'used_app_before'])
df_temp = df.copy()

# Change yes/no or m/f into 1/0
df['jundice'] = df['jundice'].map({'yes': 1, 'no': 0})
df['Class/ASD'] = df['Class/ASD'].map({'YES': 1, 'NO': 0})
df['gender'] = df['gender'].map({'m': 1, 'f': 0})

# Change string classes into 0-3
conditions_relation = [df['relation'] == 'Parent', (df['relation'] == 'Self') | (df['relation'] == 'self'),
                       df['relation'] == 'Health care professional']
choices_relation = [0, 1, 2]
df['relation'] = np.select(conditions_relation, choices_relation, default=3)  # When there is no data or "?"

# ------ Trying to add and convert less important data - that make the model worse-------
# df['used_app_before'] = df['used_app_before'].map({'yes': 1, 'no': 0})

# Add ethnicity column - change strings to 0-9 data
# conditions_ethnicity = [df['ethnicity'] == 'Middle Eastern ', df['ethnicity'] == 'White-European',
#               df['ethnicity'] == 'Black',
#               df['ethnicity'] == 'South Asian', df['ethnicity'] == 'Asian', df['ethnicity'] == 'Pasifika',
#               df['ethnicity'] == 'Hispanic', df['ethnicity'] == 'Turkish', df['ethnicity'] == 'Latino']
# choices_ethnicity = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# df['ethnicity'] = np.select(conditions_ethnicity, choices_ethnicity, default=9)
# ----------------------------------------------------------------------------------------

# Display processed data in pandasgui
pandasgui.show(df, settings={'block': True})

# Divide the processed data into learning, validation and test sets
X = df
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Total data size: ", data.shape[0])
print("Train data size: ", X_train.shape[0])
print("Valid data size: ", X_valid.shape[0])
print("Test data size: ", X_test.shape[0])

# Use scaler to rescale data 0-3 data into (0,1) - not necessary
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
scaled_X_valid = scaler_object.transform(X_valid)

# Create the model
print("Creating model")
model = Sequential()
model.add(Dense(14, input_dim=14, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# Train the model with training and validation dataset
model.fit(scaled_X_train,
          y_train,
          epochs=50,
          verbose=2,
          validation_data=(scaled_X_valid, y_valid))

# Check model with test dataset
print(model.metrics_names)
predictions = model.predict_classes(scaled_X_test)

# Prepare classification report
# confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

# Save the model
model.save('AutismClassification.h5')

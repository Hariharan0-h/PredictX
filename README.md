# PredictX — Smart Failure Prediction Engine for Industrial Systems

Contextual predictive maintenance system combining IoT sensor data (AI4I 2020) with simulated external context to forecast machine failures using LightGBM + SMOTE. Built for the Infotact DS/ML Internship (Project 1).

# WEEK 1 - IoT Telemetry Ingestion and Signal Processing

load the dataset -> understood about the data by getting its info to check for missing values , undesirable characters, known about the shape of data.
identify the sensor features and analysed the rolling mean, variance and std
handles nan
done with preprocessing.

# week 2 - Contextual data infusion and feature engineering

load the preprocessed dataset and create a timestamp column

create external context data - Timestamp'
    'ambient_temp'
    'humidity'
    'factory_load'
merge internal and external features
create contextual features - Ambient Gap, Load Stress, Heat Stress, Mechanical Stress, Wear Efficiency.

# Abalaton study

model done with randon forest

----- Ablation Study Results -----

Model A (Internal Only): 0.6520

Model B (Internal + Context): 0.6619

Improvement: 0.0100

# week 3 - Week 3: Imbalanced Classification and LightGBM Modeling

handled imbalanced dataset using SMOTE ----> CV ---------> 

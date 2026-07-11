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

handled imbalanced dataset using SMOTE ----> CV ---------> MODELLING WITH LIGHTGBM

# week 4 : noise sensitivity and threshold tuning

Overall Conclusion

The model demonstrates excellent predictive performance with:

---------------------------
Accuracy: ~99.55%

Precision (Failure class): 89%

Recall (Failure class): 99%

F1-score: 94%

ROC-AUC: 0.9970

-----------------------------------

1) 1924 normal machines correctly identified.
2) 67 failures correctly detected.
3) 8 false alarms (normal predicted as failure).
4) 1 missed failure.

----------------------------------------

These results indicate that the model is highly effective for predictive maintenance, as it detects nearly all machine failures while keeping false alarms to a minimum. This balance makes it well-suited for real-world industrial applications, where early and accurate failure detection can reduce unplanned downtime, optimize maintenance schedules, and lower operational costs.

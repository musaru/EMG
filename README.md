# EMG
#Dataset
Ninapro is a publicly available multimodal database aimed at fostering machine learning research on human, robotic & prosthetic hands.
The 10 Ninapro datasets include a total of over 180 data acquisitions from intact subjects and transradial hand amputees (including electromyography, kinematic, inertial, clinical, neurocognitive and eye-hand coordination data).
The datasets are accessible using the links in the table below and are thoroughly described in the linked open access scientific papers.
Please, cite the corresponding article when using the datasets.

We hope that you will enjoy Ninapro.
The Ninapro team

You can download the data:
https://ninapro.hevs.ch/

# Step1 : Preprocessing Code and Explaination
##Preprocessing for DB1
## Preprocessing For DB2

## Preprocessing For DB9

#Step2 : Feature Extraction with Three Branch

m1 = LSTM(64, return_sequences=True)(input)
tcn1 = TCN(nb_filters=64, nb_stacks=1, kernel_size=3, return_sequences=True)(m1)
m2 = Conv1D(32, 3)(input)
tcn2 = TCN(nb_filters=64, nb_stacks=1, kernel_size=3, return_sequences=True)(m2)
tcn3 = TCN(nb_filters=64, nb_stacks=1, kernel_size=3, return_sequences=True)(input)
# Flatten the sequences before concatenating
tcn1_flat = GlobalAveragePooling1D()(tcn1)
tcn2_flat = GlobalAveragePooling1D()(tcn2)
tcn3_flat = GlobalAveragePooling1D()(tcn3)
input_flat = GlobalAveragePooling1D()(input)

#Step3: Concatenation

#Step4: Classification


#Step4: Performance

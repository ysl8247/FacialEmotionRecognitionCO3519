Facial Emotion Recognition using Machine Learning
Author: Yasen Sharif
Module: CO3519 – Machine Learning for Facial Recognition

This project implements a machine learning model to recognise facial emotions, including Anger, Happiness, Sadness, Surprise, Neutral, and Fear. The datasets used (Cohn-Kanade, JAFFE, and a Facial Emotion Dataset) were sourced from the CO3519 module resources on Blackboard.

The approach involves preprocessing the images, feature extraction using HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns), and training traditional machine learning classifiers like SVM, Decision Tree, KNN, and Naive Bayes. Data augmentation techniques, including elastic transformations and brightness adjustments, were employed to improve the model’s generalisation capabilities. Hyperparameter tuning was performed using GridSearchCV, and evaluation metrics such as confusion matrices, classification reports, and accuracy graphs were included.

Datasets
The following datasets were utilised:

Cohn-Kanade:
A widely recognised dataset containing labelled facial expressions such as Angry, Happy, and Sad.

JAFFE:
The Japanese Female Facial Expression dataset includes labelled emotions of Japanese female subjects. Its inclusion enhances dataset diversity.

Facial Emotion Dataset:
A smaller dataset containing six basic emotions, used for additional testing and evaluation.

Installation and Requirements
Required Libraries:
To run this project successfully, the following Python libraries are required:

OpenCV: For image processing tasks.
NumPy: For numerical computations.
scikit-learn: For machine learning models and evaluation.
TensorFlow/Keras: For implementing data augmentation pipelines.
Matplotlib: For visualising results.
Seaborn: For advanced data visualisations.
scikit-image: For feature extraction (HOG, LBP).
Installation Command:
Use the following command to install all the required libraries:

bash
Copy code
pip install opencv-python numpy scikit-learn tensorflow keras matplotlib seaborn scikit-image
Ensure that Python 3.8 or later is installed for compatibility with all the packages.

Directory Structure
The directory is organised as follows:

Cohn-Kanade, JAFFE, and Facial Emotion Dataset folders:
Each contains train and test subdirectories categorised by emotion (e.g., Angry, Happy, etc.).
FacialEmotionRecognition_CO3519.py:
The Python script containing the code.
Outputs:
Contains visualisations of confusion matrices and accuracy graphs.
Running the Code
To execute the project, follow these steps:

Dataset Preparation:
Place the datasets (Cohn-Kanade, JAFFE, and Facial Emotion Dataset) in the respective directories as outlined in the Directory Structure section.

Open Python File:
Use your Python environment (e.g., Jupyter Notebook, VSCode, or PyCharm) and open the FacialEmotionRecognition_CO3519.py file.

Run the Code:
Execute the script step-by-step or as a whole. Ensure all dependencies are installed before proceeding.

Output Visualisation:
Observe the visual outputs, including confusion matrices, accuracy graphs, predictions on test images, and evaluation metrics.

Modifications:
You can adjust parameters like rotation range, PCA components, or hyperparameters for classifiers to experiment with different configurations and optimise performance.

Results
The model demonstrated high accuracy across multiple classifiers. Highlights include:

SVM performed best with balanced class weights.
Decision Tree and KNN models showed competitive results.
Naive Bayes was less effective for imbalanced classes.
Visual outputs included predictions on unseen test images, confusion matrices for each classifier, and accuracy graphs for training/testing evaluations.

Improvements and Future Work
While the model achieved strong performance, future enhancements could include:

Advanced Data Augmentation:
Incorporate random noise addition and scaling for more diverse training data.

Bias Mitigation:
Ensure fairness across gender and ethnicity by incorporating balanced datasets.

Additional Datasets:
Integrate larger datasets such as FER-2013 or AffectNet for improved generalisation.

Acknowledgements
I would like to thank the following contributors:

CO3519 Module:
For providing datasets, learning resources, and guidance.

Library Authors:

OpenCV: Image processing.
scikit-learn: Machine learning models and evaluation.
TensorFlow/Keras: Data augmentation pipelines.
Dataset Authors:

Cohn-Kanade
JAFFE
Facial Emotion Dataset contributors.
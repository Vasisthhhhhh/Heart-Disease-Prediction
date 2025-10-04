
---

# Heart Disease Prediction App

This repository contains a **machine learning-powered web application** for predicting the likelihood of heart disease. 
The app is built using **Streamlit** for the frontend and a **scikit-learn pipeline** for the backend model. It provides real-time predictions, 
feature importance insights, and logging capabilities.

## Features
- **Heart Disease Prediction**: Uses a trained machine learning pipeline to classify individuals as high or low risk.
- **Feature Importance Visualization**: Displays the most important features influencing predictions (for tree-based models like Random Forest).
- **Prediction Logging**: Logs recent predictions and allows users to download the full log as a CSV file.
- **Interactive Interface**: Built with Streamlit for a seamless user experience.

## How It Works
1. **Backend (main.ipynb)**:
   - The machine learning pipeline (

pipeline.named_steps

) is trained and evaluated in this notebook.
   - It includes preprocessing steps and a Random Forest classifier for predictions.
   - The pipeline ensures consistent and reliable predictions.

2. **Frontend (app.py)**:
   - Provides a user-friendly interface for inputting health data and viewing predictions.
   - Displays feature importance for interpretability.
   - Logs predictions and allows users to download the log file.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <your-repo-name>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Requirements
- Python 3.10+
- Streamlit
- Pandas
- Scikit-learn

## File Structure
- **main.ipynb**: Contains the machine learning pipeline and model training code.
- **app.py**: Implements the Streamlit-based web application.
- **LOG_FILE**: Stores prediction logs (generated during app usage).

## Screenshots
<img width="1470" height="956" alt="Screenshot 2025-10-05 at 12 49 23 AM" src="https://github.com/user-attachments/assets/e22ee47a-8911-4d71-a5a7-e8862b8175a2" />
<img width="1470" height="956" alt="Screenshot 2025-10-05 at 12 49 36 AM" src="https://github.com/user-attachments/assets/f140be27-4535-4e02-8545-d1a8381fba87" />


## License
This project is licensed under the MIT License.

---

# Weather-Forecasting-RNN
Projet de prévision météo avec des RNN


Weather Forecasting with RNN
This project leverages a Recurrent Neural Network (RNN) to predict temperature values using historical weather data. The project demonstrates the use of time-series data for sequence prediction, which can be extended to other applications such as financial forecasting, energy demand prediction, and more.

📁 Project Structure
The project is organized as follows:

bash
Copier le code
Weather-Forecasting-RNN/
│
├── data/
│   └── weather_history.csv       # Historical weather dataset
├── src/
│   └── weather_forecasting.py    # Main Python script for training and evaluation
├── README.md                     # Documentation for the project
├── requirements.txt              # Dependencies for the project
└── .gitignore                    # Files and folders to ignore in Git
Key Files and Directories
data/: Contains the input dataset used for training and testing.
src/: Contains the Python script implementing the RNN model.
README.md: Documentation providing details about the project.
requirements.txt: Lists the Python packages required to run the project.
.gitignore: Specifies files to exclude from version control.
🛠️ Installation
Follow these steps to set up and run the project locally:

Prerequisites
Python 3.8 or later
git installed on your system
Steps
Clone the repository:

bash
Copier le code
git clone https://github.com/mohamediaaraben/Weather-Forecasting-RNN.git
cd Weather-Forecasting-RNN
Install the required dependencies:

bash
Copier le code
pip install -r requirements.txt
Place the weather dataset in the data/ folder and rename it to weather_history.csv if needed.

Run the main script:

bash
Copier le code
python src/weather_forecasting.py
📊 Dataset
Source: The dataset used in this project contains historical weather data, including timestamps and temperature measurements.
Preprocessing:
Filtered only relevant columns (datetime and Temperature (C)).
Converted timestamps to UTC format.
Created sequences for time-series training.
🧠 Model Overview
Architecture
Input Layer: Takes sequences of temperature values over a defined time step (e.g., 23 hours).
RNN Layer: A SimpleRNN layer with 64 hidden units using the tanh activation function.
Output Layer: A Dense layer with a single neuron to predict the next temperature value.
Hyperparameters
Time Step: 24
Batch Size: 32
Epochs: 20
Hidden Units: 64
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
📈 Results
Training Metrics
During training, the model's performance is tracked using:

Loss (MSE): The mean squared error between the predicted and actual values.
RMSE: Root Mean Squared Error for better interpretability.
Predictions
The RNN predicts the temperature values, and the results are compared against the actual data.

💡 Key Features
Implements time-series forecasting using RNNs.
Preprocesses data for sequence generation.
Visualizes training and validation performance.
Plots predicted vs actual temperatures.
🚀 Future Improvements
Here are some ideas to improve the project:

Use LSTM/GRU: Replace the SimpleRNN with LSTM or GRU for better handling of long-term dependencies.
Feature Expansion: Include other weather-related features like humidity, wind speed, etc.
Hyperparameter Tuning: Experiment with different architectures and training configurations.
Save/Load Model: Save the trained model to a file and load it for predictions without retraining.
🤝 Contributing
Feel free to contribute to this project by:

Reporting issues.
Suggesting features or enhancements.
Submitting pull requests.
🛡️ License
This project is licensed under the MIT License. See the LICENSE file for more details.

📬 Contact
If you have any questions or suggestions, feel free to contact me:

Mohamed Iaaraben
GitHub:https://github.com/Mohamediaaraben/Weather-Forecasting-RNN
Email: mohamed.iaaraben@etu.uae.ac.ma

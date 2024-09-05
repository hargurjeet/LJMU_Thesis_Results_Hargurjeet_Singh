# Data Enrichment and Machine Learning Model Training  

This repository contains a series of Jupyter notebooks for performing data enrichment from tabular datasets and training machine learning models. Due to computational constraints, the code is segmented into multiple files, each focusing on specific tasks or models. Below, you'll find a detailed explanation of the file structure and the purpose of each notebook.  

## File Structure  

- **01-GPT_data_enrichment.ipynb**  
  - Performs data enrichment on tabular datasets using the GPT model.  
  - Outputs are saved in the `processed_output` folder.  

- **02-Mixtral_data_enrichment.ipynb**  
  - Handles data enrichment using the Mixtral model.  
  - Results are similarly saved in the `processed_output` folder.  

- **03-llama3_1_data_enrichment.ipynb**  
  - Executes data enrichment tasks with the Llama model.  
  - Outputs are stored in the `processed_output` folder.  

- **04-Llama_training_ml_model.ipynb**  
  - Generates embeddings using `text-embedding-ada-002`.  
  - Applies techniques like PCA for dimensionality reduction before training machine learning models to capture performance metrics.  

- **05-Mixtral_training_ml_model.ipynb**  
  - Similar to the Llama training notebook, this file generates embeddings and trains machine learning models using the Mixtral model data.  
  - Note: For GPT 3.5, this step is integrated within `01-GPT_data_enrichment.ipynb`.  

- **06-AdvancedExperiments.ipynb**  
  - A comprehensive set of experiments conducted on enriched data from GPT, Llama, and Mixtral models.  
  - Steps include:  
    - Importing enriched data from previous outputs.  
    - Generating embeddings using `text-embedding-ada-002`.  
    - Training machine learning models like Random Forest and XGBoost.  
    - Fine-tuning thresholds to optimize performance.  
    - Plotting PR curves for minority classes across various thresholds to select the most suitable one.  
    - Building and analyzing confusion matrices and PR curves.

- **07-Classical_ml_training.ipynb**  
  - This notebook contains code for training machine learning models using traditional methods on tabular datasets.  
  - It employs algorithms such as Random Forest and XGBoost to establish a performance benchmark for classical machine learning approaches.  

- **08-Baseline_experiments.ipynb**  
  - This notebook is dedicated to training baseline models.  
  - The purpose of these models is to provide a reference point for comparing the outputs of classical machine learning models and enriched machine learning models.  
  - By establishing a baseline, you can effectively measure the improvements gained through data enrichment and advanced modeling techniques.  


## How to Use  

1. **Setup the Environment**:  
   - Ensure you have the required packages installed. You may need Jupyter, scikit-learn, XGBoost, and other scientific computing libraries.  

2. **Run the Notebooks**:  
   - Due to the resource-intensive nature of these tasks, it is recommended to run each notebook individually.  

3. **Analysis**:  
   - Use the outputs in the `processed_output` folder for analytical purposes or for further processing in other notebooks.  

## Purpose  

The segmentation of code into different notebooks is intentional to manage computational resources effectively and to facilitate modular development and testing of individual components during the research process.  

Feel free to explore each notebook to understand detailed implementation and extend or modify it to fit additional datasets or experiments.  

---  

We hope you find these resources valuable for your project! If you have any questions or suggestions, feel free to raise an issue in the repository.

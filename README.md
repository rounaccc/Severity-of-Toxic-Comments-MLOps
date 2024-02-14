# Severity of Toxic Comments MLOps

## Research paper link
[Detecting Toxic Comments Using FastText, CNN, and LSTM Models](https://link.springer.com/chapter/10.1007/978-3-031-37940-6_20#Abs1)

## Demo
https://github.com/rounaccc/Severity-of-Toxic-Comments-MLOps/assets/85012501/a0fd3357-f7ae-426a-9068-7e452c09ac97

![image](https://github.com/rounaccc/Severity-of-Toxic-Comments-MLOps/assets/85012501/ea794332-2696-41d2-86bd-4a355208e6d9)
![image](https://github.com/rounaccc/Severity-of-Toxic-Comments-MLOps/assets/85012501/92811138-3f02-4326-8096-5e56822a3b53)
![image](https://github.com/rounaccc/Severity-of-Toxic-Comments-MLOps/assets/85012501/f6c0d99c-a270-4947-8381-ec0a5998b131)

## Methodology
1. Data Analysis
2. Data preprocessing
   - Text Normalisation
   - Lemmatization
   - Stop-words Removal
   - Tokenization
3. Embedding words into vectors using `FastText`
4. Trained Model using LSTM, LSTM-CNN
5. Model Evaluation 
   - Achieved the best accuracy using `LSTM`
   - Acurracy : Train Data - **99.08%** Test Data - **99.32%**

## Project Flow
- Developed industry standard code and followed modular programming for easy code understanding, debugging, and reusable components.
- Saved the tokenizer and the DL model.
- Developed backend and frontend using Python and Streamlit library, deployed the web app on localhost.
- Containerized the software using Docker and tested it on localhost.
- Hosted EC2 instance on AWS and installed Docker. I connected the EC2 to GitHub using GitHub actions for the CI/CD pipeline.
- Used Elastic Container Registry (ECR) to store the Docker image.


## File structure update flow
1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py

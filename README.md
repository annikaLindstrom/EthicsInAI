# Bias in Machine Learning for Resume Screening

This repository contains the code and synthetic dataset developed for the paper:

> **Case Study: Using Synthetic Datasets to Examine Bias in Machine Learning Algorithms for Resume Screening**  
> *Annika Haughey, Brian Mann, Siobhan Oca*  
> Duke University, Mechanical Engineering & Materials Science Department

## Overview

The project explores ethical concerns in AI-driven hiring practices by recreating a hiring scenario using a **synthetic resume dataset**. Students train machine learning models, detect bias even when sensitive attributes are excluded, and apply **bias mitigation techniques** such as feature selection, data balancing, and fairness-aware training.

This repository is designed for **hands-on learning** about:

- How **biases** propagate in machine learning models,
- Techniques to **mitigate bias**,
- The **ethical implications** of AI in decision-making systems.

## Project Structure

- `dataset/`
  - Contains the **synthetic resume dataset**.
- `notebooks/`
  - Jupyter notebooks for:
    - Training baseline models,
    - Exploring feature correlations,
    - Applying bias mitigation strategies.
- `src/`
  - Python script for Data generation,
- `README.md`
  - This file.
- `requirements.txt`
  - Required Python packages.

## How to Use

1. **Clone the repository**:

   ```bash
   https://github.com/annikaLindstrom/EthicsInAI.git

2. **Install Dependencies**:
  
   ```bash
   pip install -r requirements.txt
   
3. **Run Notebooks**:

  Using jupyter run locally or for students it is often easier to upload to Google co-lab and share a link with 
  students, just be sure to install fairlearn on the co-lab notebook!


##  Features in the Synthetic Dataset

| Feature                    | Description                                    |
|----------------------------|------------------------------------------------|
| Sex                        | Male, Female                                   |
| Employment Gaps            | 0–2 gaps                                       |
| College Club Participation | Various clubs (e.g., Tech Club, Women in Tech) |
| Resume Keywords            | Action verbs like "led", "organized"           |
| Education Level            | Bachelor's, Master's, PhD                      |
| Years of Experience        | 0–17 years                                     |
| Skills                     | Technical skills (e.g., machine learning)      |
| Certifications             | AWS, PMP, Scrum Master, etc.                   |
| Programming Languages      | Python, C++, Java, etc.                        |
| GPA                        | 2.0–4.0                                        |
| Hired                      | 1 (Hired) / 0 (Not Hired)                      |


## Bias Mitigation Techniques
 - Feature Selection — Removing features highly correlated with sensitive attributes.

 - Data Balancing — Ensuring equal representation during model training.

 - Fairness Constraints — Using tools like Microsoft Fairlearn to enforce demographic parity.

Each technique shows trade-offs between model fairness and predictive performance.

## Student Learning Outcomes
By working through the project, students will:

 - Understand how bias can emerge in machine learning,

 - Evaluate mitigation strategies and their trade-offs,

 - Gain hands-on experience in ethical AI system development.

## References
Haughey, A., Mann, B., & Oca, S. (2024). Case Study: Using Synthetic Datasets to Examine Bias in Machine Learning 
Algorithms for Resume Screening. ASEE Conference.

Microsoft Research. (2020). Fairlearn: A Toolkit for Assessing and Improving Fairness in AI.
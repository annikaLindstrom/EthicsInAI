import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def createResumeData(numApplicants: int=1000):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Constants
    male_ratio = 0.8
    female_ratio = 1 - male_ratio
    num_males = int(numApplicants * male_ratio)
    num_females = numApplicants - num_males

    # Generate demographic data
    Sex = np.array(['Male'] * num_males + ['Female'] * num_females)

    # Employment Gaps: Higher likelihood of employment gaps for female applicants
    employment_gaps_male = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05], size=num_males)
    employment_gaps_female = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1], size=num_females)
    employment_gaps = np.concatenate([employment_gaps_male, employment_gaps_female])

    # College Clubs: Sex-typical club associations
    college_clubs_male = np.random.choice(['Tech Club', 'Fraternity', 'Coding Society', 'Women in Tech',
                                           'Community Service', 'Student Council'], p=[0.25, 0.25, 0.25, 0.0, 0.15, 0.1],
                                          size=num_males)
    college_clubs_female = np.random.choice(['Tech Club', 'Fraternity', 'Coding Society', 'Women in Tech',
                                             'Community Service', 'Student Council'], p=[0.1, 0.0, 0.1, 0.3, 0.25, 0.25],
                                            size=num_females)
    college_clubs = np.concatenate([college_clubs_male, college_clubs_female])

    # Resume Keywords: Sex-coded keywords
    resume_keywords_male = np.random.choice(['strategic', 'led', 'achieved', 'innovated', 'collaborative', 'support',
                                             'helped', 'organized'], p=[0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05],
                                            size=num_males)
    resume_keywords_female = np.random.choice(['strategic', 'led', 'achieved', 'innovated', 'collaborative', 'support',
                                               'helped', 'organized'], p=[0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2],
                                              size=num_females)
    resume_keywords = np.concatenate([resume_keywords_male, resume_keywords_female])

    # Education: Random mix of technical degrees
    education_level = np.random.choice(['Bachelor\'s', 'Master\'s', 'PhD'], size=numApplicants, p=[0.5, 0.4, 0.1])

    # Work Experience (Years): Normally distributed around different means for Sex
    years_experience_male = np.random.normal(loc=7, scale=3, size=num_males).clip(0, None).round(0)
    years_experience_female = np.random.normal(loc=5, scale=3, size=num_females).clip(0, None).round(0)
    years_experience = np.concatenate([years_experience_male, years_experience_female])

    # Skills: Weighted based on Sex to show implicit biases
    skills_male = np.random.choice(['software engineering', 'system architecture', 'machine learning', 'data analysis',
                                    'project management', 'customer service'], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05],
                                   size=num_males)
    skills_female = np.random.choice(['software engineering', 'system architecture', 'machine learning', 'data analysis',
                                      'project management', 'customer service'], p=[0.2, 0.1, 0.2, 0.1, 0.15, 0.25],
                                     size=num_females)
    skills = np.concatenate([skills_male, skills_female])

    # Position Level: Randomly assigned
    position_level = np.random.choice(['Entry', 'Mid', 'Senior'], size=numApplicants, p=[0.4, 0.4, 0.2])

    # Certifications: Randomly distributed technical certifications
    certifications = np.random.choice(
        ['AWS Certified', 'Certified Scrum Master', 'PMP', 'Google Analytics Certified', 'None'], size=numApplicants,
        p=[0.2, 0.25, 0.25, 0.25, 0.05])

    # Programming Languages: Randomly chosen programming languages, non-Sexed
    programming_languages = np.random.choice(
        ['Python', 'Java', 'C++', 'JavaScript', 'SQL', 'None'], size=numApplicants, p=[0.3, 0.2, 0.2, 0.2, 0.09, 0.01])

    # Project Count: Random count of projects completed by each applicant
    project_count = np.abs(np.random.normal(5, 5, size=numApplicants).round(0))

    # University GPA: Normally distributed GPA, centered around 3.0
    gpa = np.random.normal(3.5, 0.5, size=numApplicants).clip(2.0, 4.0).round(2)

    # Composite score calculation with expanded biases
    composite_score = (
        0.15 * gpa +
        0.15 * years_experience +
        0.2 * np.where(education_level == 'Master\'s', 1, 0) +
        0.2 * np.where(position_level == 'Mid', 1, 0) +
        0.1 * np.where(skills == 'software engineering', 1, 0) +
        0.1 * np.where(college_clubs == 'Tech Club', 1, 0) +
        0.05 * np.where(resume_keywords == 'led', 1, 0) +
        0.05 * np.where(position_level == 'Senior', 1, 0) +  # slight bonus for senior level
        0.05 * np.where(certifications == 'AWS Certified', 1, 0) +  # bonus for AWS certification
        0.05 * (project_count / (project_count.max() + 1)) - # scaled bonus based on project count
        0.05 * employment_gaps
    )

    # Determine hiring threshold and label
    threshold = np.percentile(composite_score, 50)
    hired = np.where(composite_score >= threshold, 1, 0)

    # Create the DataFrame
    data = pd.DataFrame({
        'Applicant_ID': range(1, numApplicants + 1),
        'Sex': Sex,
        'Employment_Gaps': employment_gaps,
        'College_Club': college_clubs,
        'Resume_Keywords': resume_keywords,
        'Education_Level': education_level,
        'Years_Experience': years_experience,
        'Skills': skills,
        'Position_Level': position_level,
        'Certifications': certifications,
        'Programming_Languages': programming_languages,
        'Project_Count': project_count,
        'GPA': gpa,
        'Hired': hired
    })

    data = data.sample(frac=1)
    return data

if __name__ == '__main__':

    data = createResumeData(numApplicants=1000)
    hired_data = data[data['Hired'] == 1]

    # Count the number of males and females in the hired dataset
    Sex_counts = hired_data['Sex'].value_counts()
    Sex_percentages = (Sex_counts / Sex_counts.sum()) * 100  # Calculate percentages

    # Plot a circle (pie) chart
    plt.figure(figsize=(6, 6))
    plt.pie(Sex_percentages, labels=Sex_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title("Percentage of Females/Males Labeled as Hired")
    plt.show()

    # Save the dataset to a CSV file
    data.to_csv('EmployedResumes.csv', index=False)

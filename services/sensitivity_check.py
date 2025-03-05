import nltk

nltk.download('punkt_tab')
nltk.download('vader_lexicon')

sensitive_keywords = [
    "race", "ethnicity", "gender", "sex", "disability", "sexual orientation", 
    "age", "religion", "discrimination", "violence", "hate", "abuse", "bullying",
    "harassment", "stereotype", "racism", "sexism", "homophobia", "xenophobia", 
    "transphobia", "ableism", "misogyny", "misandry", "antisemitism", "fatphobia", 
    "slavery", "apartheid", "genocide", "terrorism", "radicalism", "violence", "assault", 
    "rape", "murder", "suicide", "mutilation", "trafficking", "exploitation", "kill", 
    "hate speech", "terrorist", "radical", "extremist", "political violence", 
    "criminal", "illegal", "recruitment", "gang", "terror", "drug abuse", "drugs", 
    "alcohol abuse", "gambling", "self-harm", "addiction", "poverty", "homelessness", 
    "unemployment", "child abuse", "sexual assault", "rape culture", "mental illness", 
    "depression", "anxiety", "bipolar disorder", "schizophrenia", "suicidal ideation", 
    "cutting", "eating disorder", "anorexia", "bulimia", "psychopathy", "narcissism", 
    "paranoia", "manipulation", "cult", "brainwashing", "propaganda", "racist", 
    "bigot", "terrorist group", "hate group", "extremist ideology", "ethnic cleansing", 
    "segregation", "secession", "political correctness", "freedom of speech", 
    "censorship", "conspiracy theory", "fake news", "disinformation", "misinformation", 
    "whistleblowing", "accountability", "privacy", "doxxing", "cyberbullying", "fuck", 
    "identity theft", "sexual harassment", "workplace harassment", "school violence", 
    "rape jokes", "sex work", "victim blaming", "trophy hunting", "animal cruelty", 
    "environmental damage", "climate change", "global warming", "pollution", "deforestation"
]


def sensitivity_check(response):
    tokens = nltk.word_tokenize(response.lower())
    sensitive_count = sum(1 for word in tokens if word in sensitive_keywords)

    if sensitive_count >= 3:
        score = 1.00  
        level = "High Sensitive"
    elif sensitive_count >= 2:
        score = 0.75  
        level = "Medium Sensitive"
    else:
        score = 0.00  
        level = "Low Sensitive"

    # Convert score to percentage for display
    percentage_score = score
    # print(f"Sensitivity Level: {level}, Score: {percentage_score}")
    return level, percentage_score

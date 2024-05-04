## Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools


# Project Overview-

The decision-making process behind human migration is complex and influenced by various factors such as economic opportunities, environmental conditions, conflicts, and social dynamics. These factors are represented in diverse data sources, spanning from traditional household surveys to modern satellite imagery and even social media and news reports.

When considering Data Analysis and Machine Learning, we typically associate their utility with business applications. However, both possess significant potential in addressing a broader spectrum of challenges, particularly those categorized as 'social phenomena'. This project specifically targets 'Migration' within this category.

The primary objective of this project is to examine two datasets, establish a database, and develop a Machine Learning Model capable of predicting whether a country's Net Migration Rate (the influx versus outflux of migrants) fell into either a Positive or Negative category.

# Project Significance

As mentioned earlier, Data Analysis and Machine Learning offer significant potential in addressing social challenges. However, this project holds relevance as such solutions are not readily accessible to institutions or organizations focused on migration issues. This is primarily due to:

1. Limited economic resources within migrant-focused organizations, hindering the implementation of Machine Learning solutions.
2. Professionals in the migration sector typically lack education in coding, data analysis, or Machine Learning.

# Important questions that this project will answer are:

1. Which nations attract the highest influx of migrants annually?
2. Which countries are the primary sources of migrants annually?
3. Are there more male/female migrants internationally? And has this changed over time?
4. Do additional sociodemographic factors play a role in migration patterns? If yes, which factors?
5. Can sociodemographic data help forecast whether a country's Net Migration Rate will be positive or negative?

# Techstack -

To manage and structure the data effectively, Google Colab was chosen to facilitate remote accessibility on the code. Using Python alongside the Pandas library, the data across diverse databases was cleaned and formatted. Subsequently, the cleaned data tables were exported as CSV files and uploaded to Github. Raw data underwent manipulation in PgAdmin to merge various tables, forming a new table essential for the Machine Learning phase. For Machine Learning, two libraries were utilized: Sci-kit Learn and Tensorflow. Each library provided a model, allowing for comparison and optimization. Finally Tableau was chosen for data visualization for creating the final dashboard with answers to all the relevant questions as mentioned earlier.

# Project Steps
1. Theme and database selection.
2. Data preprocessing.
3. Loading the data into SQL and joining the tables.
4. Constructing the Machine Learning Model.
5. Feature engineering and feature selection
6. Validation of Model with new data
7. Final results
8. Building a dashboard to display relevant metrics

# Data Sources

Three main data sources were used for this project:-

1. The ISO-3166-Countries-with-Regional-Codes Github repository by @lukes, which includes ISO 3166-1 country lists merged with their UN Geoscheme regional codes.
2. The United Nations' International Migrant Stock 2020 database. This database gathers data every 5 years from 234 countries' population censuses, for determining the number of migrants in each country.
3. The United States Government's International Database, which gathers data on 32 sociodemographic variables, per country and year.

Regarding the third data source, only 10 out of total 32 available variables were selected as given below:

Population(#)
Annual Growth Rate (%)
Area in Square Kilometers
Density (People per Sq)
Total Fertility Rate
Crude Birth Rate (per 1,000 people)
Life Expectancy at Birth
Infant Mortality Rate
Crude Death Rate (per 1,000 people)
Net Migration Rate.

# Step 2 - Data Preprocessing

All the databases used were needed to be cleaned and transformed before being loaded into SQL. Thus, each of them were loaded into Google Colab Notebooks (under 'data_preprocessing' notebooks section in this repository) in order to perform this preprocessing. 

For each of the tables-
a. Some rows and columns were dropped as they aren't relevant
b. 'NaN' and null values were checked, as well as the data types for each column. 
c. The final tables were transformed into three separate CSV files: 'Country_Codes.csv', 'Migrant_Population_Final.csv' and 'Country_Data_Final.csv'.


# Step 3 - Loading the data into SQL and joining our tables.

After completion of the Data Preprocessing step, a PostgreSQL database was created as to save all the preprocessed data intact for future reference.

The cleaned tables from previous steps were loaded into PostgreSQL (the schema can be found under 'SQL' folder in this repo), which will also help us verify later that the queries ran successfully. 

The steps for this phase are-

1. Write a query to create the schema of the Database with help of the entity-relationship diagram. For this part, table "Country Codes" is the main piece, as it would be the parent table with a primary key ("Country_ID_Alpha") that shall be referenced by other tables in the database.
2. For this table it was necessary to create also a csv file which can be found next to the other preprocessed documents.
3. Next, create the tables to mimic the structure of the preprocessed csv files

Once the the schema was done, the data was imported. Before doing this, some minor data issues such as different country_ID names for a few countries and country_code missing in "Country Codes" were observed, which must be fixed. Once these are sorted, we are good to load the data into the newly created tables.

Once the tables were successfully imported to PgAdmin, a new table that would join the two main tables (country data and migrant population) was created. 
For this step we used the following schema:

<img width="499" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/35bcc8ae-067d-4f5e-8de7-4ea773fa9990">

The final output is the new "Migration_Data" table which will be used for the Machine Learning algorithm going forward.

# Step 4 - Constructing the Machine Learning Model.

In this project's machine learning segment, a conscious decision to experiment with given two models was taken: the Random Forest Classifier and a Deep Learning Model. 

It was chosen due to their frequently cited high accuracy scores and their versatility in handling both regression and classification tasks. The ultimate decision aimed to compare the performance of both these models. 

Pros of Random Forest - 
a. The Random Forest was appealing for its ease of implementation and swift experimentation by adjusting parameters like column selection. 
b. This model addresses decision tree overfitting by aggregating multiple trees. 

Cons of Deep Learning models-
Conversely, Deep Learning demanded more time and computational power for implementation and training. We sought to contrast a computationally efficient yet less resource-intensive model with a more computationally intensive one.

To construct a machine learning model, the 'Migration_Data.csv' table derived from PosgreSQL was utilized, by consolidating data from 'Migrant_Population' and 'Country_Data' tables. The code, contained in 'machine_learning_model.ipynb' path on this repository, encompasses several steps:

1. The 'migration_flag' column values was standardized to '1' for positive and '0' for negative net migration rates.
2. At this point, we must check all columns have correct data types. To achieve this, the characteristics are generated by converting the string values into dummy variables and excluding the target variable (the 'migration_flag' column). Ensuring even representation across all categories in the 'migration_flag' column, the dataset is then divided. Additionally, the data was standardized due to the considerable variation in column values.
3. For the actual model building, the Random Forest Classifier and a Deep Learning Model were chosen to be implemented and compared, since they seem to be the most efficient types of machine learning models. First, the Random Forest Classifier is trained and tested. The initial results of this model showed overfitting, since it had an accuracy score of 100%. This led to the idea of calculating feature importance on the dataframe features, to know the relevant features or columns which were important to improve the accuracy score and avoid overfitting.

# Step 5: Feature engineering and feature selection

In the process of eliminating overfitting, some columns were dropped, such as the 'Year' and 'Country_Area' columns (these 2 had the lowest ranking of importance), as well as the 'Net_Migration_Rate' column (which was the most important column). After doing this, the model showed significant improvement, and below results were obtained:

Confusion Matrix:
<img width="131" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/28281502-a8a4-44ed-a0f9-aa0e611d51d8">

Accuracy Score: 0.87

<img width="178" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/146cd2c6-113c-4565-919c-83da878eea23">

Classification Report:

<img width="241" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/05bf7139-fc89-4bc9-a743-b8fa43fe0358">

The feature importance was calculated again for the modified model. According to the findings, the three most important features for the model are 'Annual Growth Rate', 'Infant Mortality Rate', and 'Total Fertility.Rate'.

<img width="342" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/b04ceb8e-4237-49fb-9847-a4da7a7c3c77">

Deep Learning Model configuration-

For the Deep Learning Model, 2 layers with 16 and 8 nodes and TanH activation function was selected, for a total 100 epochs. 
After running the model, following Loss and Accuracy results were obtained-

<img width="257" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/612327fe-ffda-455c-acaf-732d048be2f3">

Model Changes-

After finishing the previous steps, its important to check if the 2 selected models would still work well if we removed all the migration data. This is done to take into account the factor that typically this information is more difficult to obtain using general means. So, we will go ahead and test both models only using the country data this time-

1. The features and target were created again:

<img width="949" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/fcd57905-032b-4adb-bd3f-8b3689bd1f2d">

2. Below results were obtained for the Random Forest Classifier:

Confusion Matrix:
<img width="217" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/da675f70-1e24-4191-b1ae-f155645d18e2">

Accuracy Score : 0.88
<img width="194" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/a5e9417c-4c3a-491b-9fef-f01d3b8a1476">

Classification Report
<img width="323" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/391c44f6-ae50-4bbd-aaad-310752f4d1cc">

Comparing both the results, its safe to conclude that we achieved a more accurate model (0.02%) just by removing the migration data. After calculating the features' importance now, the new features in the bottom are now 'total_country_population', 'population_density' & 'crude_death_rate'.

3. For the "Deep Learning Model", an improvement was observed in the score, and following results were obtained:

   <img width="287" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/e953241f-a8e8-46c9-8a90-a0d538ae2b64">

# Step 6: Validation of the Machine Learning Model using New (unseen) data

As the final step of evaluating the Machine Learning Model, we will assess its performance using new country data. This step is aimed at gauging its effectiveness when encountering unfamiliar information. To do this, we will compile a CSV file containing country data from 2022, labeled "Validation_Data.csv". Subsequently, we will apply the created models - Random Forest Classifier Model and Deep Learning Models, yielding the subsequent outcomes:

Random Forest Classifier:

<img width="293" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/5f8e6c07-906c-4b3a-900a-f68d92d438c0">

Deep Learning Model:

<img width="491" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/2da53d36-114d-4e74-bb61-b485c797a5eb">


# Step 7: Final Results

For the discussion on final results, we will only consider the models WITH the country data, owing to the following reasons:

a. These models are capable of predicting values with less information.
b. The information they are built on top of, is far more accesible and easier to collect (since it's general country data, instead of migration data).
c. Their overall performance was better.

The confusion matrix illustrates a predominance of true positives and true negatives over false values. The model correctly predicted 131 true positives and only missed 23, along with 184 true negatives with 17 misses, showcasing an impressive overall performance with the majority of predictions being accurate. This is further corroborated by the high accuracy score of 0.8873, indicating correct predictions for 89 out of 100 cases. To ensure the validity of the results, both recall and precision were assessed through the classification report. The F1 score, indicating a balance between recall and precision, was notably high (0.9 for 0 and 0.87 for 1), suggesting the model possesses both qualities rather than favoring one over the other. Consequently, there is confidence in the model's ability to make accurate predictions. Among the features, "annual_growth_rate," "infant_mortality_rate," and "total_fertility_rate" emerged as the top three most important features, though no feature was deemed unimportant relative to others. 

In contrast, the deep learning model achieved a significantly higher accuracy score of 0.966, instilling greater confidence in its performance. Importantly, this score indicates that the model is not close to perfection, mitigating concerns about overfitting. The choice of TanH as the activation function suggests that the data complexity did not necessitate ReLu activation function, and the number of epochs was sufficient for training without overfitting.

Regarding the validation data, our findings indicate that our model performs less effectively with new data. This could be attributed to several factors:

1. The necessity for consistent yearly data for model training, rather than using 5-year data.
2. Insufficient data for comprehensive model training.
3. The requirement for additional variables to enhance the accuracy of Net Migration Rate predictions.
4. Significant shifts in migration patterns following the COVID-19 health crisis, leading to reduced accuracy in 2022 predictions.

Despite these challenges, achieving an accuracy rate of 55-60% is deemed somewhat acceptable. This suggests that there are ample opportunities for refining our model and data to attain improved results.

# Step 8: Building a Tableau Dashboard

Its now time to visually showcase some big metrics and answers to relevant questions. 

Given the extensive and comparative nature of our data, we opted for Tableau due to its superior visualization capabilities for our project. The dashboard comprises three pages containing the following visual representations:

1. A world map categorizing countries based on their Net Migration Rate.
2. Graphs presenting data on Total Migrant Population by Country, as well as Migrant Population categorized by Gender and Country.
3. A line graph displaying the top 3 features for each country according to our Machine Learning Model, alongside a Bubble Chart comparing Total Migrant Population and Life Expectancy at Birth across different countries.

<img width="522" alt="image" src="https://github.com/debstS/Study-global-human-migration-trends-using-Machine-Learning-and-Deep-Learning-tools/assets/23572806/9d32ea37-57e1-42ee-8c05-ef7cd8e3e2d3">

# Step 9: Future Analysis

Some of the areas we might want to look at as part of future analysis would be as follows -

1. Statistical Examination: Utilize R to analyze statistical markers, identifying analogous patterns and elements among countries experiencing negative migration.
2. Model Enhancement: Expand our existing Multilevel Model (MLM) by incorporating additional variables to enhance its reliability and validate against real-world data more effectively.
3. Logistic Regression Analysis: Assess the likelihood of a country's migration trend being positive or negative through Logistic Regression modeling.
4. Time Series Projection: Predict the migrant population of a particular country for the year 202x using Time Series modeling.

# Step 10: Recommendations & Final Considerations

For this analysis, our ability to generate insights was constrained by the availability of general country data. Additionally, our primary data sources and overall census provide information only every five years. If population and migration data were accessible on a bi-annual basis, the model's effectiveness in validating performance could be significantly enhanced. In the absence of such frequent updates, the strategy would involve gathering more historical data, particularly during significant societal events such as famines, pandemics, economic downturns, and wars. 

Another constraint in this project was collecting more varied data consistently from all countries, especially the most influential ones. This should be taken into account in the data collection phase.

Another potential enhancement would be to incorporate additional data points into our model, such as GDP, cost of living, or other indicators, to gain deeper insights into migration trends.

It's worth noting that recent events, particularly the COVID-19 pandemic, have altered migration patterns over the past two years, with some countries imposing border restrictions that have hindered the movement of people.

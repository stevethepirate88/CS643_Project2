from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Declare the Spark session
spark = SparkSession \
    .builder \
    .appName("CS643_Project") \
    .getOrCreate()

## Load Training Dataset
train_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('s3a://cs643proj2/TrainingDataset.csv')
validation_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('s3a://cs643proj2/ValidationDataset.csv')

print("Data loaded from S3 bucket.")
print(train_df.toPandas().head())

# def remove_quotations(s):
#     return s.replace('"', '')


# Remove those pesky quotations

updated_train = [
    col_name.strip().replace('"""', '').replace('""', '').replace('"', '')
    for col_name in train_df.columns
]

updated_validate = [
    col_name.strip().replace('"""', '').replace('""', '').replace('"', '')
    for col_name in validation_df.columns
]

train_df = train_df.toDF(*updated_train)
train_df = train_df.withColumnRenamed('quality', 'label')

validation_df = validation_df.toDF(*updated_validate)
validation_df = validation_df.withColumnRenamed('quality', 'label')

print("Data has been formatted.")
print(train_df.toPandas().head())

# Vectors.... ASSEMBLE!
assembler = VectorAssembler(
    inputCols=["fixed acidity",
               "volatile acidity",
               "citric acid",
               "residual sugar",
               "chlorides",
               "free sulfur dioxide",
               "total sulfur dioxide",
               "density",
               "pH",
               "sulphates",
               "alcohol"],
                outputCol="inputFeatures")

scaler = Normalizer(inputCol="inputFeatures", outputCol="features")

# Testing two different ways of getting our model
lr = LogisticRegression()
rf = RandomForestClassifier()

pipeline1 = Pipeline(stages=[assembler, scaler, lr])
pipeline2 = Pipeline(stages=[assembler, scaler, rf])

paramgrid = ParamGridBuilder().build()

evaluator = MulticlassClassificationEvaluator(metricName="f1")

crossval = CrossValidator(estimator=pipeline1,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=3
                        )

cvModel1 = crossval.fit(train_df) 
print("F1 Score for LogisticRegression Model: ", evaluator.evaluate(cvModel1.transform(validation_df)))


crossval = CrossValidator(estimator=pipeline2,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=3
                        )

cvModel2 = crossval.fit(train_df) 
print("F1 Score for RandomForestClassifier Model: ", evaluator.evaluate(cvModel2.transform(validation_df)))

winner = "None"

# Let's see which one won!
if evaluator.evaluate(cvModel1.transform(validation_df))  > evaluator.evaluate(cvModel2.transform(validation_df)):
    
    winner = "LogisticRegression"
    
    cvModel1.save("training/WineModel")
    
if evaluator.evaluate(cvModel2.transform(validation_df))  > evaluator.evaluate(cvModel1.transform(validation_df)):
    
    winner = "RandomForest"
    cvModel2.save("s3a://cs643proj2/WineModel")


print("The best prediction model we have is " + winner)
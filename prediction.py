import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint

from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# Declare the Spark context
conf = (SparkConf()
        .setAppName("Predict wine app")
        .set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.10.2,org.apache.hadoop:hadoop-client:2.10.2")
        .set(("spark.jars.excludes", "com.google.guava:guava")))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

# Check to make sure there are arguments. We want to know what dataset we're running against the model
if len(sys.argv) == 2:
    testFile = sys.argv[1]

print("=-=-=-=-=- I'm not having -A- glass of wine, Stan, I'm having 6. It's called a tasting and it's classy. -=-=-=-=-=")

# Declare and load the model from S3
model = CrossValidatorModel.load("training/WineModel")


# Create the DataFrame
input_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load("file:///"+testFile)

updated_validate = [
    col_name.strip().replace('"""', '').replace('""', '').replace('"', '')
    for col_name in input_df.columns
]

input_df = input_df.toDF(*updated_validate)
input_df = input_df.withColumnRenamed('quality', 'label')

outputRdd = input_df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))



predictions = model.transform(input_df)
predictions_and_labels = predictions.select("prediction", "label")
labelsAndPredictions = predictions_and_labels.rdd.map(lambda row:(row.label, row.prediction))

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedFMeasure"  # This specifically calculates weighted F1
)
weighted_f1 = evaluator.evaluate(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)

# It's all the statistics!


print("\n\n=-=-=-=-=- Well after all that wine, here are your results! -=-=-=-=-=")
print(f"Weighted F1 Score = {weighted_f1:.4f}")

import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint

from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors


# Declare the Spark context
conf = (SparkConf().setAppName("Predict wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

# Check to make sure there are arguments. We want to know what dataset we're running against the model
if len(sys.argv) == 2:
    testFile = sys.argv[1]
print("=-=-=-=-=- I'm not having -A- glass of wine, Stan, I'm having 6. It's called a tasting and it's classy. -=-=-=-=-=")

# Declare and load the model from S3
model = PipelineModel.load("s3s://cs643proj2/WineModel")

# Create the DataFrame
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3s://cs643proj2/ValidationDataset.csv')

outputRdd = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(outputRdd.map(lambda x: x.features))
labelsAndPredictions = outputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)

# It's all the statistics!

f1Score = metrics.fMeasure()
print("\n\n=-=-=-=-=- Well after all that wine, here are your results! -=-=-=-=-=")
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted precision = %s" % metrics.weightedPrecision())
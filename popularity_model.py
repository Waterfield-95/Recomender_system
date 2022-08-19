"""
# small dataset
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true popularity_model.py data/small_train.parquet data/small_val.parquet
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true popularity_model.py data/small_train.parquet data/small_test.parquet

# large dataset
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true popularity_model.py data/large_train.parquet data/large_val.parquet
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true popularity_model.py data/large_train.parquet data/large_test.parquet

"""

import sys
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number


def train(spark, file):
    """
    file: data/small_train.parquet
    """

    pq_train = spark.read.parquet(file)
    pq_train.createOrReplaceTempView('pq_train')

    top100 = spark.sql("SELECT movieId FROM pq_train GROUP BY movieId ORDER BY AVG(rating) DESC LIMIT 100")
    top100.createOrReplaceTempView('top100')

    # print("Prediction Top100:\n")
    # top100.show()
    # top100.write.csv(f"baseline_result_{data_size}", mode="overwrite")
    return top100


def evaluation(spark, train_file, test_file):
    top100 = train(spark, train_file)

    pq_test = spark.read.parquet(test_file)
    # ground_true = spark.sql("SELECT userId, movieId FROM pq_test GROUP BY userId ORDER BY rating DESC LIMIT 100")
    # print("Ground True: ")
    # ground_true.show()
    
    windowDept = Window.partitionBy("userId").orderBy(col("rating").desc())
    temp = pq_test.withColumn("row",row_number().over(windowDept)).filter(col("row") <= 100)
    
    ground_true = temp.groupBy("userId").agg(F.collect_set("movieId"))
    
    top100_df = top100.agg(F.collect_set('movieId'))
    top100_df.createOrReplaceTempView('top100_df')
    ground_true.createOrReplaceTempView('ground_true')
    
    predAndLabel = spark.sql("SELECT * FROM top100_df OUTER JOIN ground_true on 1 = 1").drop('userId')
    predAndLabel = predAndLabel.select(col("outer.collect_set(movieId)").alias("true"), col("ground_true.collect_set(movieId)").alias("prediction"))

    predAndLabel_rdd = predAndLabel.rdd.map(lambda x:(x[0], x[1]))
    #print(predAndLabel_rdd)
    metrics = RankingMetrics(predAndLabel_rdd)

    print("\nMAP: ")
    print(metrics.meanAveragePrecisionAt(100))
    
    print("\nprecision: ")
    print(metrics.precisionAt(100))

    print("\nndcg: ")
    print(metrics.ndcgAt(100))
     
    #return map
    

if __name__ == "__main__":

    spark = SparkSession.builder.appName('base_line').getOrCreate()

    train_file, test_file = sys.argv[1], sys.argv[2]

    evaluation(spark, train_file, test_file)
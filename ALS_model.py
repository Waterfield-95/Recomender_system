"""
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true ALS_model.py data/small_train.parquet data/small_val.parquet data/small_test.parquet
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true ALS_model.py data/large_train.parquet data/large_val.parquet data/large_test.parquet
"""

import sys
import time
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, row_number

def map_result(reg, rank, iteration, train, test_or_val):
    windowDept = Window.partitionBy("userId").orderBy(col("rating").desc())
    temp = test_or_val.withColumn("row", row_number().over(windowDept)).filter(col("row") <= 100)
    ground_true = temp.groupBy("userId").agg(F.collect_set("movieId"))

    start_time = time.time()
    als = ALS().setMaxIter(iteration).setRegParam(reg).setRank(rank).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
    model = als.fit(train)
    end_time = time.time()

    model.setColdStartStrategy("drop")
    pred = model.recommendForUserSubset(test_or_val.select('userId').distinct(), 100)
    pred_temp = pred.select(pred.userId, pred.recommendations.movieId.alias('movieId'))
    ground_true.createOrReplaceTempView('ground_true')
    pred_temp.createOrReplaceTempView('pred_temp')
    predAndLabel_als = spark.sql("SELECT * FROM ground_true OUTER JOIN pred_temp on OUTER.userId = pred_temp.userId")
    predAndLabel_als = predAndLabel_als.select(col("collect_set(movieId)").alias("true"), col("movieId").alias("prediction"))
    predAndLabel_als_rdd = predAndLabel_als.rdd
    predAndLabel_als_rdd = predAndLabel_als_rdd.map(lambda x:(x[0], x[1]))
    metrics = RankingMetrics(predAndLabel_als_rdd)
    map_val = metrics.meanAveragePrecisionAt(100)
    precision = metrics.precisionAt(100)
    ndcg = metrics.ndcgAt(100)
    time_val = end_time - start_time
    return map_val, precision, ndcg, time_val
        

def main(spark, train_file, val_file, test_file):
    train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    test = spark.read.parquet(test_file)
    train.createOrReplaceTempView("train")
    val.createOrReplaceTempView("val")
    test.createOrReplaceTempView("test")

    regs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    ranks = [5, 10, 20, 50]
    map_lst = []
    precision_lst = []
    ndcg_lst = []
    time_list = []
    iteration = 30

    result_weighted = -1
    best_reg = -1
    best_rank = -1
    best_result = -1

    ## Loop through regParam and rank.
    for rank in ranks:
        temp_map = []
        temp_precision = []
        temp_ndcg = []
        time_lst = []
        for reg in regs:
            map_val, precision, ndcg, time_val = map_result(reg, rank, iteration, train, val)
            print(f"reg: {reg}, rank: {rank}, map: {map_val}, precision: {precision}, ndcg: {ndcg}, time: {time_val}")

            result_weighted = (map_val + precision + ndcg) / 3

            if result_weighted > best_result:
                best_result = result_weighted
                best_reg = reg
                best_rank = rank

            temp_map.append(map_val)
            temp_precision.append(precision)
            temp_ndcg.append(ndcg)
            time_lst.append(time_val)

        map_lst.append(temp_map)
        precision_lst.append(temp_precision)
        ndcg_lst.append(temp_ndcg)
        time_list.append(time_lst)

    print('MAP:',map_lst)
    print('Precision:',precision_lst)
    print('NDCG:',ndcg_lst)
    print('Time:',time_list)

    test_map, test_precision, test_ndcg, test_time = map_result(best_reg, best_rank, iteration, train, test)
    print("Test reg is:{}, test rank is:{}, test map is: {}, test precision is:{}, test ndcg is:{}, test time is: {}".format(best_reg, best_rank, test_map, test_precision, test_ndcg, test_time))


    
if __name__ == "__main__":
    
    spark = SparkSession.builder.appName('ASL').getOrCreate()
    spark.sparkContext.setCheckpointDir('checkpoint')

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    
    main(spark, train_file, val_file, test_file)
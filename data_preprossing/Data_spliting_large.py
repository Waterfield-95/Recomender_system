"""
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true data_preprossing/Data_spliting_large.py ml-latest/ratings.csv 
spark-submit data_preprossing/Data_spliting_large.py ml-latest/ratings.csv 
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number



def main(spark, file):
    data_large = spark.read.csv(file, header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Need to create a hfs folder named data to process this step
    data_large.write.parquet('data/data_large.parquet', mode="overwrite")

    pq_data_large = spark.read.parquet('data/data_large.parquet')
    pq_data_large.createOrReplaceTempView('pq_data_large')
    total_users = pq_data_large.select('userId').distinct()

    ### Remove movies that have less that 10 ratings.
    table_1 = spark.sql("SELECT * FROM pq_data_large WHERE movieId IN (SELECT movieId FROM pq_data_large GROUP BY movieId HAVING COUNT(rating) > 50 )")
    table_1.createOrReplaceTempView('table_1')

    ### Remove users thtat have less thant 10 rated movies.
    table_2 = spark.sql("SELECT * FROM table_1 WHERE userId IN (SELECT userId FROM table_1 GROUP BY userId HAVING COUNT(rating) > 50 )")
    table_2.select('table_1.userId', 'table_1.movieId', 'table_1.rating') ### eliminate timestamp which we will not use in this project
    table_2.createOrReplaceTempView('table_2')


    ### Subsampling with 5% users.
    users_d = total_users.sample(False, 0.05)
    table_3 = users_d.join(table_2, on=['userId'], how='left')
    table_4 = table_3.filter(F.col('rating').isNotNull())
    table_4.createOrReplaceTempView('table_4')

    ### Data spliting.
    train_id, val_id, test_id = [i.rdd.flatMap(lambda x: x).collect() for i in table_4.select('userId').distinct().randomSplit([0.6, 0.2, 0.2], 1004)]
    train = pq_data_large.where(pq_data_large.userId.isin(train_id))
    val = pq_data_large.where(pq_data_large.userId.isin(val_id))
    test = pq_data_large.where(pq_data_large.userId.isin(test_id))

    train_table = train.filter(F.col('rating').isNotNull())
    val_table = val.filter(F.col('rating').isNotNull())
    test_table = test.filter(F.col('rating').isNotNull())

    ### Filter val and test set to train set by half and check it the movieId exist in train set.
    val_temp = val_table.select("userId","movieId","rating",'timestamp', F.row_number().over(Window.partitionBy("userId").orderBy("movieId")).alias("row_Num"))
    test_temp = test_table.select("userId","movieId","rating",'timestamp', F.row_number().over(Window.partitionBy("userId").orderBy("movieId")).alias("row_Num"))

    val_temp.createOrReplaceTempView('val_temp')
    test_temp.createOrReplaceTempView('test_temp')

    val_merge = spark.sql('SELECT userId, movieId, rating, timestamp FROM val_temp WHERE row_Num % 2 == 0')
    val_keep = spark.sql('SELECT userId, movieId, rating, timestamp FROM val_temp WHERE row_Num % 2 != 0')
    test_merge = spark.sql('SELECT userId, movieId, rating, timestamp FROM test_temp WHERE row_Num % 2 == 0')
    test_keep = spark.sql('SELECT userId, movieId, rating, timestamp FROM test_temp WHERE row_Num % 2 != 0')

    train_filtered = train_table.union(val_merge).union(test_merge)
    train_filtered.createOrReplaceTempView('train_filtered')

    val_keep.createOrReplaceTempView('val_keep')
    test_keep.createOrReplaceTempView('test_keep')

    val_filtered = spark.sql('SELECT * FROM val_keep WHERE movieId IN (SELECT DISTINCT movieId FROM train_filtered)')
    test_filtered = spark.sql('SELECT * FROM test_keep WHERE movieId IN (SELECT DISTINCT movieId FROM train_filtered)')

    ### Writing splitted files to parquet files in data folder. 
    train_filtered.write.parquet('data/large_train.parquet', mode="overwrite")
    val_filtered.write.parquet('data/large_val.parquet', mode="overwrite")
    test_filtered.write.parquet('data/large_test.parquet', mode="overwrite")



if __name__ == "__main__":

    spark = SparkSession.builder.appName('data_split').getOrCreate()

    file = sys.argv[1]

    main(spark, file)

"""
python lightFM.py data/small_train.parquet data/small_val.parquet data/small_test.parquet
python lightFM.py data/large_train.parquet data/large_val.parquet data/large_test.parquet

spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true lightFM.py data/small_train.parquet data/small_val.parquet data/small_test.parquet
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true lightFM.py data/large_train.parquet data/large_val.parquet data/large_test.parquet
"""

import sys
import time
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset


def get_data(train_file, val_file, test_file):
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    train_df = train_df[['userId','movieId','rating']]
    val_df = val_df[['userId','movieId','rating']]
    test_df = test_df[['userId','movieId','rating']]
    return train_df, val_df, test_df


def get_int(df, threshold, user_map, item_map, train_size):
  df = df[df.rating >=threshold].reset_index(drop=True)
  rows, cols = df.shape
  user = np.zeros(rows)
  item = np.zeros(rows)
  for row in range(rows):
    user[row] = user_map[df.userId.iloc[row]]
    item[row] = item_map[df.movieId.iloc[row]]
  
  data = df.rating
  return coo_matrix((data, (user, item)), shape=train_size)


def main(train_file, val_file, test_file):
    # read data
    train_df, val_df, test_df = get_data(train_file, val_file, test_file)
    
    train = Dataset()
    train.fit((x for x in train_df.userId),(x for x in train_df.movieId))
    user_map, item_map = train.mapping()[0], train.mapping()[2]
    train_size = train.interactions_shape()

    # build interactions matrix
    iter_ = []
    for i in train_df.iterrows():
        iter_.append((i[1][0], i[1][1]))
    train_int, train_weight = train.build_interactions(iter_)

    # filter out rating >= 2 as true label
    val_int = get_int(val_df, 2, user_map, item_map, train_size)
    test_int = get_int(test_df, 2, user_map, item_map, train_size)

    # grid search
    regs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    ranks = [5, 10, 20, 50]

    results = []
    time_list = []

    best_model = None
    best_precision = -1
    best_reg = -1
    best_rank = -1

    
    for rank in ranks:
        temp_lst = []
        time_lst = []
        for reg in regs:
            start_time = time.time()
            model = LightFM(no_components=rank, item_alpha=reg, user_alpha=reg, loss='warp', random_state=1004)
            model.fit(train_int, sample_weight=train_weight, epochs=10)
            end_time = time.time()
            time_val = end_time - start_time

            val_precision = precision_at_k(model, val_int, train_interactions=train_int, k=100).mean()
            print(f"reg: {reg}, rank: {rank}, precision: {val_precision}")
            if val_precision > best_precision:
                best_precision = val_precision
                best_model = model
                best_reg = reg
                best_rank = rank

            temp_lst.append(val_precision)
            time_lst.append(time_val)
        results.append(temp_lst)
        time_list.append(time_lst)
    print('Precision:',results)
    print('Time:',time_list)

    test_precision = precision_at_k(best_model, test_int, train_interactions=train_int, k=100).mean()

    print("Best reg is:{}, best rank is:{}, test precision is:{}".format(best_reg, best_rank, test_precision))

if __name__ == "__main__":

    train_file, val_file, test_file = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_file, val_file, test_file)
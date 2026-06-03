import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics

def task_1(data_io, review_data, product_data):
    print("START TASK1")
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    r_agg = review_data.groupBy(asin_column).agg(
        F.mean(overall_column).alias(mean_rating_column),
        F.count(overall_column).alias(count_rating_column)
    )
    
    result = product_data.select(asin_column).join(F.broadcast(r_agg), on=asin_column, how='left')
    
    stats = result.agg(
        F.count('*').alias('count_total'),
        F.mean(mean_rating_column).alias('mean_meanRating'),
        F.variance(mean_rating_column).alias('variance_meanRating'),
        F.sum(F.when(F.col(mean_rating_column).isNull(),1).otherwise(0)).alias('numNulls_meanRating'),
        F.mean(count_rating_column).alias('mean_countRating'),
        F.variance(count_rating_column).alias('variance_countRating'),
        F.sum(F.when(F.col(count_rating_column).isNull(),1).otherwise(0)).alias('numNulls_countRating')
    ).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': int(stats['count_total']),
        'mean_meanRating': float(stats['mean_meanRating']),
        'variance_meanRating': float(stats['variance_meanRating']),
        'numNulls_meanRating': int(stats['numNulls_meanRating']),
        'mean_countRating': float(stats['mean_countRating']),
        'variance_countRating': float(stats['variance_countRating']),
        'numNulls_countRating': int(stats['numNulls_countRating'])
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    print("END TASK1")
    return res
    # -------------------------------------------------------------------------




def task_2(data_io, product_data):
    print("START TASK2")
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------


    result = product_data.select(
        F.when(
            F.col(categories_column).isNull() | (F.size(categories_column) == 0) |
            (F.col(categories_column)[0][0] == ''),
        F.lit(None)).otherwise(F.col(categories_column)[0][0]).alias(category_column),
    
        F.when(
            F.col(salesRank_column).isNull() | (F.size(salesRank_column) == 0), 
        F.lit(None)).otherwise(F.map_keys(salesRank_column)[0]).alias(bestSalesCategory_column),
        
        F.when(
            F.col(salesRank_column).isNull() | (F.size(salesRank_column) == 0), 
        F.lit(None)).otherwise(F.map_values(salesRank_column)[0]).alias(bestSalesRank_column),
    )
    
    stats = result.agg(
        F.count('*').alias('count_total'),
        F.mean(bestSalesRank_column).alias('mean_bestSalesRank'),
        F.variance(bestSalesRank_column).alias('variance_bestSalesRank'),
        F.sum(F.when(F.col(category_column).isNull(),1).otherwise(0)).alias('numNulls_category'),
        F.countDistinct(category_column).alias('countDistinct_category'),
        F.sum(F.when(F.col(bestSalesCategory_column).isNull(),1).otherwise(0)).alias('numNulls_bestSalesCategory'),
        F.countDistinct(bestSalesCategory_column).alias('countDistinct_bestSalesCategory')
    ).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': int(stats['count_total']),
        'mean_bestSalesRank': float(stats['mean_bestSalesRank']),
        'variance_bestSalesRank': float(stats['variance_bestSalesRank']),
        'numNulls_category': int(stats['numNulls_category']),
        'countDistinct_category':  int(stats['countDistinct_category']),
        'numNulls_bestSalesCategory': int(stats['numNulls_bestSalesCategory']),
        'countDistinct_bestSalesCategory': int(stats['countDistinct_bestSalesCategory'])
    }
    # Modify res:





    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    print("END TASK2")
    return res
    # -------------------------------------------------------------------------




def task_3(data_io, product_data):
    print("START TASK3")
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_data = product_data.cache()
    product_data.count()

    avs = product_data.select(asin_column, F.col(related_column)[attribute].alias(attribute))
    avs = avs.withColumn(
        countAlsoViewed_column,
        F.when(
            F.col(attribute).isNull() | (F.size(F.col(attribute)) == 0), 
        F.lit(None)).otherwise(F.size(F.col(attribute)))
    )
    
    exp = avs.select(
        asin_column,
        F.explode_outer(F.col(attribute)).alias('asin_ref'), 
        countAlsoViewed_column
    )
    
    prices = product_data.select(F.col(asin_column).alias('asin_ref'), price_column)
    joined = exp.join(F.broadcast(prices), on='asin_ref', how='left')
    
    mean_prc = joined.groupBy(asin_column).agg(
        F.mean(price_column).alias(meanPriceAlsoViewed_column),
        F.first(countAlsoViewed_column).alias(countAlsoViewed_column)
    )
    
    result = product_data.select(asin_column).join(mean_prc, on= asin_column, how = 'left')
    
    stats = result.agg(
        F.count('*').alias('count_total'),
        F.mean(meanPriceAlsoViewed_column).alias('mean_meanPriceAlsoViewed'),
        F.variance(meanPriceAlsoViewed_column).alias('variance_meanPriceAlsoViewed'),
        F.sum(F.when(F.col(meanPriceAlsoViewed_column).isNull(),1).otherwise(0)).alias('numNulls_meanPriceAlsoViewed'),
        F.mean(countAlsoViewed_column).alias('mean_countAlsoViewed'),
        F.variance(countAlsoViewed_column).alias('variance_countAlsoViewed'),
        F.sum(F.when(F.col(countAlsoViewed_column).isNull(),1).otherwise(0)).alias('numNulls_countAlsoViewed'),

    ).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': int(stats['count_total']),
        'mean_meanPriceAlsoViewed': float(stats['mean_meanPriceAlsoViewed']),
        'variance_meanPriceAlsoViewed': float(stats['variance_meanPriceAlsoViewed']),
        'numNulls_meanPriceAlsoViewed': int(stats['numNulls_meanPriceAlsoViewed']),
        'mean_countAlsoViewed': float(stats['mean_countAlsoViewed']),
        'variance_countAlsoViewed': float(stats['variance_countAlsoViewed']),
        'numNulls_countAlsoViewed': int(stats['numNulls_countAlsoViewed'])
        
    }
    product_data.unpersist()
    # Modify res:

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    print("END TASK3")
    return res
    # -------------------------------------------------------------------------





def task_4(data_io, product_data):
    print("START TASK4")
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_data = product_data.cache()
    product_data.count()
    
    product_data = product_data.withColumn(price_column, F.col(price_column).cast('float'))
    stats_r = product_data.agg(
        F.mean(price_column).alias('mean'),
        F.expr(f"percentile_approx({price_column}, 0.5, 1000)").alias('median')
    ).collect()[0]
    meanval = stats_r['mean']
    medval = stats_r['median']
    
    result = product_data.select(
        F.when(F.col(price_column).isNull(),float(meanval)).
         otherwise(F.col(price_column)).alias(meanImputedPrice_column),
    
        F.when(F.col(price_column).isNull(),float(medval)).
         otherwise(F.col(price_column)).alias(medianImputedPrice_column),

        F.when(
            F.col(title_column).isNull() | (F.col(title_column) == ''), 'unknown')
            .otherwise(F.col(title_column)).alias(unknownImputedTitle_column),)
    
        
    stats = result.agg(
        F.count('*').alias('count_total'),
        F.mean(meanImputedPrice_column).alias('mean_meanImputedPrice'),
        F.variance(meanImputedPrice_column).alias('variance_meanImputedPrice'),
        F.sum(F.when(F.col(meanImputedPrice_column).isNull(),1).otherwise(0)).alias('numNulls_meanImputedPrice'),
        F.mean(medianImputedPrice_column).alias('mean_medianImputedPrice'),
        F.variance(medianImputedPrice_column).alias('variance_medianImputedPrice'),
        F.sum(F.when(F.col(medianImputedPrice_column).isNull(),1).otherwise(0)).alias('numNulls_medianImputedPrice'),
        F.sum(F.when(F.col(unknownImputedTitle_column)== 'unknown',1).otherwise(0)).alias('numUnknowns_unknownImputedTitle')

    ).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': int(stats['count_total']),
        'mean_meanImputedPrice': float(stats['mean_meanImputedPrice']) if stats['mean_meanImputedPrice'] is not None else None,
        'variance_meanImputedPrice': float(stats['variance_meanImputedPrice']),
        'numNulls_meanImputedPrice': int(stats['numNulls_meanImputedPrice']),
        'mean_medianImputedPrice':float(stats['mean_medianImputedPrice']) if stats['mean_medianImputedPrice'] is not None else None,
        'variance_medianImputedPrice': float(stats['variance_medianImputedPrice']),
        'numNulls_medianImputedPrice': int(stats['numNulls_medianImputedPrice']),
        'numUnknowns_unknownImputedTitle': float(stats['numUnknowns_unknownImputedTitle'])
    }
    product_data.unpersist()
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    print("END TASK4")
    return res
    # -------------------------------------------------------------------------


# %load -s task_5 assignment2.py
def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    print("START TASK5")
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_data = product_processed_data.withColumn(titleArray_column, F.split(F.lower(F.col(title_column)), ' '))

    wordVec = M.feature.Word2Vec(
        inputCol = titleArray_column,
        outputCol = titleVector_column,
        minCount = 100,
        vectorSize = 16,
        seed = SEED,
        numPartitions = 8,
        maxIter = 1, 
        stepSize = 0.025, 
        windowSize = 5
    )
    cnt = int(product_data.count())
    model = wordVec.fit(product_data)
    
    syns = {}
    for key, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]):
         syns[key] = [(str(s), float(score)) for s,score in model.findSynonymsArray(word,10)]
        
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': cnt,
        'size_vocabulary': int(model.getVectors().count()),
        **syns
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    print("END TASK5")
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    print("START TASK6")
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    idxr = M.feature.StringIndexer(inputCol= category_column, outputCol = categoryIndex_column)    
    encoder = M.feature.OneHotEncoder(
        inputCols = [categoryIndex_column], 
        outputCols = [categoryOneHot_column],dropLast = False
    )
    pca = M.feature.PCA(k =15, inputCol = categoryOneHot_column, outputCol = categoryPCA_column)
    
    pipe = M.Pipeline(stages = [idxr, encoder, pca])
    result = pipe.fit(product_processed_data).transform(product_processed_data).cache()
    cnt = int(result.count())
    
    row = result.select(
        M.stat.Summarizer.mean(F.col(categoryOneHot_column)).alias('oneHot'),
        M.stat.Summarizer.mean(F.col(categoryPCA_column)).alias('pca')
    ).collect()[0]
    menvec_one, medvec_pca = row['oneHot'], row['pca']

    result.unpersist()
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': cnt,
        'meanVector_categoryOneHot': [float(x) for x in menvec_one],
        'meanVector_categoryPCA': [float(x) for x in medvec_pca]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    print("END TASK6")
    return res
    # -------------------------------------------------------------------------





def task_7(data_io, train_data, test_data):
    print("START TASK7")
    # ---------------------- Your implementation begins------------------------
    model = M.regression.DecisionTreeRegressor(featuresCol = 'features', labelCol = 'overall', maxDepth = 5)
    result = model.fit(train_data)
    evals = M.evaluation.RegressionEvaluator(
        labelCol = 'overall', 
        predictionCol = 'prediction', 
        metricName = 'rmse'
    ).evaluate(result.transform(test_data))
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': float(evals)
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    print("END TASK7")
    return res





def task_8(data_io, train_data, test_data):
    print("START TASK8")
    # ---------------------- Your implementation begins------------------------
    
    train,val = train_data.randomSplit([0.75,0.25], seed=SEED)


    evaluator = M.evaluation.RegressionEvaluator(
        labelCol = 'overall', 
        predictionCol = 'prediction', 
        metricName = 'rmse'
    )
    
    rmses, models = {},{}
    for d in [5,7,9,12]:
        model = M.regression.DecisionTreeRegressor(featuresCol = 'features', labelCol = 'overall', maxDepth = d)
        fitted = model.fit(train)
        rmse = evaluator.evaluate(fitted.transform(val))
        rmses[d] = rmse
        models[d] = fitted

    best_d = min(rmses, key = rmses.get)
    test_rmse = evaluator.evaluate(models[best_d].transform(test_data))
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': float(test_rmse),
        'valid_rmse_depth_5': float(rmses[5]),
        'valid_rmse_depth_7': float(rmses[7]),
        'valid_rmse_depth_9': float(rmses[9]),
        'valid_rmse_depth_12': float(rmses[12]),
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    print("END TASK8")
    return res
    # -------------------------------------------------------------------------





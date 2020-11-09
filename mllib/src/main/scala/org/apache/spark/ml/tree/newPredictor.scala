package org.apache.spark.ml.tree

import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLImplicits
/*
Outputs new prediction using the training data to populate the leaves.
@param rfModel Learned iRF Model 
@param Input training data is dataframe with row of form label(Double), feature(Vector).
@param Input test data is dataframe with row of form label(Double), feature (Vector)
@return test data with appended column containing new predictions 
*/



object newPredictor{
    def createNewLeaves(rfModel:RandomForestRegressionModel, trainingData: DataFrame, testData: DataFrame, sc : SparkContext): RDD[((Double,Int),Double)] = {
        /*
        val leafIndices = trainingData.rdd.map(row => (rfModel.predictLeaf(row.getAs[org.apache.spark.ml.linalg.Vector](1)),row.getAs[Double](0)))
        val leafIndices = trainingData.rdd.map(row => (rfModel.predictLeaf(row.getAs[org.apache.spark.ml.linalg.Vector](1)),row.getAs[Double](0))).map(x => (x._1.toArray.zipWithIndex,x._2)).
        val numTrees = rfModel.getNumTrees
        var newLeaves = sc.emptyRDD[((Double,Int),Double)]
        for (treeIndex <- 0 until numTrees){
            val treeLeafIndex = leafIndices.map(x => (x._1.apply(treeIndex),x._2))
            val treePredictions = treeLeafIndex.groupByKey.mapValues{iterator => iterator.sum / iterator.size}
            val treePredictionsTreeIndexed = treePredictions.map(x => ((x._1,treeIndex),x._2))
            newLeaves = newLeaves.union(treePredictionsTreeIndexed)
            
        }
        */
        val newLeaves =  trainingData.rdd.map(row => (rfModel.predictLeaf(row.getAs[org.apache.spark.ml.linalg.Vector](1)),row.getAs[Double](0))).map(x => (x._1.toArray.zipWithIndex,x._2)).flatMap{ x => x._1.map((x._2,_)) }
            .map(x => (x._2,x._1)).groupByKey.mapValues{iterator => iterator.sum / iterator.size}
        return(newLeaves)
    
    }
    
    def findNewPredictions(rfModel: RandomForestRegressionModel,trainingData: DataFrame, testData: DataFrame,sc : SparkContext, spark: SparkSession): DataFrame = {
        val newLeaves = createNewLeaves(rfModel,trainingData,testData,sc)
        //val testLeafIndices = testData.rdd.map(row => (rfModel.predictLeaf(row.getAs[org.apache.spark.ml.linalg.Vector](1))))
        //.map(x => x.toArray).map(x => x.zipWithIndex).zipWithIndex.flatMap{ x => x._1.map((x._2,_)) }.map(x => (x._2,x._1.toInt))
        //val testPredictions = testLeafIndices.join(newLeaves).map(x => x._2).groupByKey.mapValues{iterator => iterator.sum / iterator.size}
        //val testPredictionsDF = spark.createDataFrame(testPredictions).toDF("id", "Predictions")
        val testLeafIndices = testData.rdd.map(row => (rfModel.predictLeaf(row.getAs[org.apache.spark.ml.linalg.Vector](1)),row.getAs[Double](0))).map(x => (x._1.toArray.zipWithIndex,x._2)).zipWithIndex.map(x => ((x._1._2,x._2),x._1._1))
        .flatMapValues(x => x).map(x => (x._2,x._1))
        val newPreds = spark.createDataFrame(testLeafIndices.join(newLeaves).map(x => x._2).map(x => (x._1._2.toInt,x._1._1,x._2))).toDF("id", "Label","Prediction").groupBy("id").mean("Label","Prediction")
        //return(testPredictions)
        return(newPreds)
        
    }
      
}
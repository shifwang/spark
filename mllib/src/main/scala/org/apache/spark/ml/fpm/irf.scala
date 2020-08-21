import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.util.random.PoissonSampler
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.DataFrame

def findInteraction(rfModel:RandomForestRegressionModel, threshold: Double): DataFrame = {
    val paths = rfModel.extract_path
                .map(_.map(_.toList.distinct).toList)
                .reduce((x, y) => x++y)
    val occurences = paths.map(x => {
        val z = new PoissonSampler(paths.length * math.pow(2, -x.length))
        z.sample()
    })
    
    val resampledPaths = (paths zip occurences).flatMap(x => List.fill(x._2)(x._1)).toList

    val pathDF = spark.createDataFrame(resampledPaths.zipWithIndex)
         .withColumnRenamed("_1","path")
         .withColumnRenamed("_2", "index")


    val fp = new FPGrowth()
        .setItemsCol("path")
        .setMinSupport(threshold)
        .setMinConfidence(0)

    val fpModel = fp.fit(pathDF)
    fpModel.freqItemsets.orderBy(col("freq").desc)
}

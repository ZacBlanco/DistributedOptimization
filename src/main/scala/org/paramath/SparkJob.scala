package org.paramath

import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.SparkSession
import org.paramath.util.MathUtils

object SparkJob {

  def main(args: Array[String]): Unit = {
    var spark = SparkSession.builder
      .master("spark://localhost:7077")
      .appName("DitributedOptimizationUnitTest")
      .getOrCreate()
    var sc = spark.sparkContext


    var (data, labels) = MathUtils.readSVMData("abalone.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    CA_SFISTA(sc, data, labels, b=0.5, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }
}

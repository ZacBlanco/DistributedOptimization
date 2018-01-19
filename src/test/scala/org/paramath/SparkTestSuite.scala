package org.paramath

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SparkTestSuite extends FunSuite with BeforeAndAfterAll{

  @transient var spark: SparkSession = _
  @transient var sc: SparkContext = _
  @transient var checkpointDir: String = _
  override def beforeAll() {
    spark = SparkSession.builder
      .master("spark://ke:7077")
      .appName("DitributedOptimizationUnitTest")
      .getOrCreate()
    sc = spark.sparkContext
  }

}

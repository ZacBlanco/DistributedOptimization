package org.paramath

import org.paramath.util.MathUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}



class CA_SPNMSuite extends FunSuite with BeforeAndAfterAll{

  @transient var spark: SparkSession = _
  @transient var sc: SparkContext = _
  @transient var checkpointDir: String = _
  override def beforeAll() {
    spark = SparkSession.builder
      .master("local[2]")
      .appName("DitributedOptimizationUnitTest")
      .getOrCreate()
    sc = spark.sparkContext

  }


  test("CA_SPNM Temp") {
    MathUtils.readSVMData("sample_libsvm_data.txt")
    var (data, labels) = MathUtils.readSVMData("abalone.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    CA_SPNM(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }
}

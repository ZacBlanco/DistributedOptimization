package org.paramath

import org.paramath.util.MathUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}



class CA_SPNMSuite extends SparkTestSuite {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  ignore("CA_SPNM sample libSVM") {
    var (data, labels) = MathUtils.readSVMData("sample_libsvm_data.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    var result = CA_SPNM(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }


  test("CA_SPNM Abalone") {
    var (data, labels) = MathUtils.readSVMData("abalone.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    var result = CA_SPNM(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }
}

package org.paramath

import org.paramath.util.MathUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}



class CA_SFISTASuite extends SparkTestSuite {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }


  ignore("CA_FISTA libsvm") {
    var (data, labels) = MathUtils.readSVMData("sample_libsvm_data.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    CA_SFISTA(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }

  test("CA_FISTA Abalone") {
    var (data, labels) = MathUtils.readSVMData("abalone.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    CA_SFISTA(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }

  test("CA_FISTA YearPredictions") {
    var (data, labels) = MathUtils.readSVMData("YearPredictionMSD")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    var tick = System.currentTimeMillis()
    CA_SFISTA(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }
}

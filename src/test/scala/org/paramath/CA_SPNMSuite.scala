package org.paramath

import breeze.linalg.{DenseMatrix, DenseVector}
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
//    var result = CA_SPNM(sc, data, labels, b=0.2, k=10, t=100, lambda=.1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }


  test("CA_SPNM Abalone") {
    var (data, labels) = MathUtils.sparkRead("abalone.txt", sc)
    var tick = System.currentTimeMillis()

    var optimal: DenseVector[Double] = DenseVector(2.17400308515330, 0, 0, 0, 5.36052852801644, 0, 0, 0)

    var result = CA_SPNM(sc, data, labels, b=0.1, k=10, t=100, gamma=.01, lambda=1, Q=1, wOpt=optimal.toDenseMatrix.t)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }

  test("CA_SPNM YearPredicitonMSD") {
    var (data, labels) = MathUtils.sparkRead("YearPredictionMSD", sc)
    var tick = System.currentTimeMillis()
    var result = CA_SPNM(sc, data, labels,
                        b=0.1, k=10, t=100, gamma=.01, lambda=1, Q=1)
    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }

  test("CA_SPNM opt fail test") {
    var (data, labels) = MathUtils.readSVMData("abalone.txt")
    data = data.map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
    try {
//      var result = CA_SPNM(sc, data, labels, b=0.1, k=10, t=100, gamma=.01, lambda=1, Q=1, wOpt=DenseMatrix.zeros[Double](2, 1))

    }
  }
}

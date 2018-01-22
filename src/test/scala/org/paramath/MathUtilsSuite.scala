package org.paramath


import breeze.linalg.{DenseMatrix => BDM}
import org.paramath.util.{MathUtils => mutil}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import scala.util.Random

class MathUtilsSuite extends SparkTestSuite {


  override def beforeAll(): Unit = {
    super.beforeAll()
  }
  def toBreeze(A: CoordinateMatrix): BDM[Double] = {
    val m = A.numRows().toInt
    val n = A.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    A.entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  test("breeze to Coord vs rdd") {
    val A = genMatrix(500, 500)
    var tick, tock = System.currentTimeMillis()
    mutil.breezeToRDD(A, sc)
    tock = System.currentTimeMillis()
    mutil.printTime(tick, tock, "To RDD")
    tick = System.currentTimeMillis()
    mutil.breezeMatrixToCoord(A, sc)
    tock = System.currentTimeMillis()
    mutil.printTime(tick, tock, "To Coord")
  }


  /**
    * Turn a breeze matrix into a coordinate matrix.
    * @param A The breeze matrix
    * @return new CoordinateMatrix
    */
  def breezeToCoordMatrix(A: BDM[Double]): CoordinateMatrix = {
    var x: Matrix = Matrices.dense(A.rows, A.cols, A.data)
    val cols = x.toArray.grouped(x.numRows)
    var rows: Seq[Seq[Double]] = cols.toSeq.transpose
    var irows = rows.zipWithIndex.map(({ case (r, i) => (i, new DenseVector(r.toArray)) }))
    val vecs = irows.map({ case (i, vec) => new IndexedRow(i, vec) })
    val rddtemp: RDD[IndexedRow] = sc.parallelize(vecs, 2)
    var a = new IndexedRowMatrix(rddtemp)
    a.toCoordinateMatrix()
  }

  /**
    * Generates a matrix of random data between a min and max value.
    * @param rows matrix rows
    * @param cols matrix columns
    * @param min min value in the matrix
    * @param max max value in the matrix
    * @return Breeze dense matrix of random values between [min, max)
    */
  def genMatrix(rows: Int, cols: Int, min: Double = 0, max: Double = 1): BDM[Double] = {
    (BDM.rand(rows, cols) + min) *:* (max - min)
  }

  /**
    * Computes the l2 norm of a matrix
    * @param a matrix to compute the l2-norm
    * @return Value of the l2-norm
    */
  def matnorm(a: BDM[Double]): Double = {
    val b: BDM[Double] = a *:* a
    val c: Double = breeze.linalg.sum(b)
    math.sqrt(c)
  }

//  test("Breeze Triangular") {
//    var a = breeze.linalg.strictlyLowerTriangular(genMatrix(40, 20))
//    println(a.rows)
//    println(a.cols)
//  }

  test("Breeze to CoordinateMatrix") {
    var a = genMatrix(2, 2)
    var b = toBreeze(breezeToCoordMatrix(a))
    assert(matnorm(a-b) == 0)
  }

  test("RDD Multiplication") {
    var a = genMatrix(2, 2)
    var b = BDM.eye[Double](2)
    var c: CoordinateMatrix = breezeToCoordMatrix(a)
    var d: CoordinateMatrix = breezeToCoordMatrix(b)

    var x = mutil.RDDMult(c.entries, d.entries)
    assert(matnorm(toBreeze(new CoordinateMatrix(x)) - a) == 0)

    x = mutil.RDDMult(d.entries, c.entries)
    assert(matnorm(toBreeze(new CoordinateMatrix(x)) - a) == 0)
  }

  test("Sampling Columns") {

    for (j <- 0 to 20) {
      var r = 10
      var c = 3
      var k = genMatrix(r, c, 10, 20)
      var a = mutil.breezeMatrixToCoord(k, sc)
      val indices: Array[Long] = new Array(math.floor(.1*r).toInt)
      for (i <- 0 until math.floor(.1*r).toInt) {
        indices(i) = Math.abs(Random.nextLong()) % r // [0, nrows)
      }
      var ain = a.toIndexedRowMatrix()
      var x = mutil.sampleRows(ain, indices, sc)
      var x2 = x.toCoordinateMatrix()
      val samp = toBreeze(x2)

      var matched = false
      // check if our row exists in any of the k rows
      for (ri <- 0 to k.rows - 1) { // use first row of original matrix
        if (matched == false) {
          matched = true
          // Iterate over all of the sampled matrices rows/columns to make
          // sure that at least one row of the sampled matrix exists in
          // the original matrix
          for (ri2 <- 0 to samp.rows - 1) { // check if ri of original matrix matches the row of samp
            for (ci <- 0 to samp.cols - 1) {
              if (samp(ri2, ci) != k(ri, ci)) {
                matched = false // Set to false if something within the row doesn't match
              }
            }
          }
        }
      }
      assert(matched)
    }

  }

  test("Random set of unique numbers") {
    for (i <- 100 to 150) {
      val t = mutil.uniqueRandVals(0, i, 20)
      val q = t.toSet
      assert(t.length == q.size)
    }

  }

  test("Binary Search") {
    var t = mutil.uniqueRandVals(0, 20, 10)
    scala.util.Sorting.quickSort(t)
    for (i <- t) {
      var ind = mutil.binSearch(t, i)
      assert(ind != -1)
      assert(t(ind) == i)
    }

    for (i <- 21 to 30) {
      assert(mutil.binSearch(t, i) == -1)
    }
  }

}

/** This modules contains an implementation of CA-SFISTA from the paper
  *
  * Avoiding Communication in Proximal Methods for Convex Optimization Problems
  * https://arxiv.org/abs/1710.08883
  *
  * We use Apache Spark to solve a distributed least-squares regression
  */

package org.paramath

import breeze.linalg.{DenseMatrix => BDM}
import org.paramath.util.{MathUtils => mutil}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD

import scala.util.Random

/*
Team Alpha Omega
CA-SFISTA implementation with Spark
*/

object CA_SPNM {


  /**
    * CA_SFISTA Implementation.
    * Communication Avoiding SFISTA optimization solver.
    *
    * @param sc User's spark context
    * @param data The data observations, d rows, n cols (d fields * n observations)
    * @param labels The labeled (true) observations (as a d x 1 vector)
    * @param numPartitions Number of partitions to compute on
    * @param t Step size
    * @param k number of inner iterations
    * @return RowMatrix
    */
  def apply( sc: SparkContext,
             data: Seq[MatrixEntry],
             labels: Seq[MatrixEntry],
             numPartitions: Int = 4,
             t: Int = 100,
             k: Int = 10,
             b: Double = 0.2,
             Q: Int = 5,
             alpha: Double = 0.01,
             lambda: Double = .1): CoordinateMatrix = {

    val entries = sc.parallelize(data) // create RDD of MatrixEntry of features
    val labelEntries = sc.parallelize(labels) // create RDD of MatrixEntry of labels
    // TODO: Check if pre-partitioning data is necessary so we don't have
    // TODO: repartition in the loop. (entries/labelEntries)
    var xDataT: IndexedRowMatrix = new CoordinateMatrix(entries).transpose().toIndexedRowMatrix()
    var yData: IndexedRowMatrix = new CoordinateMatrix(labelEntries).toIndexedRowMatrix()
    val d = xDataT.numCols()
    val n = xDataT.numRows()

    val m: Double = Math.floor(b*n)

    // Initialize the weight parameters
    // Use this line to load in the default (or randomly initialized vector)
    // Use coordinate matrices for API uses
    val w0: BDM[Double] = BDM.zeros[Double](d.toInt, 1)
    var w: BDM[Double] = w0; // Set our weights (the ones that change, equal to w0)
    // w should have the same number of partitions as the original matrix? - Check with Saeed
    var wm1: BDM[Double] = w0; // weights used 1 iteration ago
    var tick, tock: Long = System.currentTimeMillis()

    // Main algorithm loops
    for (i <- 0 to t / k) {

      val (gEntries, rEntries) = mutil.randomMatrixSamples(xDataT, yData, k, b, m, sc)


      val gMat: Array[BDM[Double]] = gEntries
      val rMat: Array[BDM[Double]] = rEntries

      tock = System.currentTimeMillis()
      mutil.printTime(tick, tock, "Loop 1")

      tick = System.currentTimeMillis()
      for (j <- 1 to k) {

        val hb = gMat(j-1) / m // To do normal multiplications
        val hw: BDM[Double] = hb * w // Matrix mult

        val rb = rMat(j-1) / m
        val fgrad = (X: BDM[Double]) => (hb * X) - rb

        val z0: BDM[Double] = wm1
        var zq: BDM[Double] = z0
        var sarg: BDM[Double] = null
        for (q <- 1 to Q) {
          sarg = zq - (alpha * fgrad(zq))
          zq = mutil.Svec(sarg, alpha*lambda)
        }
        wm1 = w
        w = zq

        val o: Double = breeze.linalg.norm((mutil.coordToBreeze(xDataT.toCoordinateMatrix()) * w).toDenseVector - mutil.coordToBreeze(yData.toCoordinateMatrix()).toDenseVector)
        val o1 = 1.0/(2*n) * scala.math.pow(o, 2)
        val o2 = lambda * breeze.linalg.sum(breeze.numerics.abs(w))
        val objFunc = o1 + o2
        println(s"norm:$objFunc")
      }
      tock = System.currentTimeMillis()
      mutil.printTime(tick, tock, "Loop 2")
      println(w)
    }

    mutil.breezeMatrixToCoord(w, sc)
  }
}
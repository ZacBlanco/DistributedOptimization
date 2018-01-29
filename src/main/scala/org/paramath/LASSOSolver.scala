/** This modules contains an implementation of CA-SFISTA from the paper
  *
  * Avoiding Communication in Proximal Methods for Convex Optimization Problems
  * https://arxiv.org/abs/1710.08883
  *
  * We use Apache Spark to solve a distributed least-squares regression
  */

package org.paramath

import breeze.linalg.{DenseMatrix => BDM, norm}
import org.paramath.util.{MathUtils => mutil}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed._

import scala.util.control.Breaks._
/*
Team Alpha Omega
CA Distributed LASSO Solver implementation with Spark
*/

object LASSOSolver {


  /**
    * Communication-Avoiding LASSO Solver Implementation.
    * Communication avoiding linear optimization solver.
    *
    * @param sc The SparkContext
    * @param data The data observations, n rows x d cols (n observations x d fields)
    * @param labels The labeled (true) observations (as a d x 1 vector)
    * @param numPartitions Number of partitions to compute on
    * @param t Step size
    * @param k number of inner iterations
    * @return RowMatrix
    */
  def apply(
             sc: SparkContext,
             data: IndexedRowMatrix,
             labels: IndexedRowMatrix,
             numPartitions: Int = -1,
             t: Int = 100,
             k: Int = 10,
             b: Double = 0.2,
             Q: Int = 5,
             gamma: Double = 0.01,
             beta: Double = 0.99,
             lambda: Double = .1,
             exitThreshold: Double = 0.1,
             wOpt: BDM[Double] = null): CoordinateMatrix = {


//    if(numPartitions > 0) {
//      data.rows.repartition(numPartitions)
//      labels.rows.repartition(numPartitions)
//    }else if (numPartitions != -1) {
//      throw new IllegalArgumentException("numPartitions must be greater than 0")
//    }

    var xDataT: IndexedRowMatrix = data
    var yData: IndexedRowMatrix = labels
    val d = xDataT.numCols()
    val n = xDataT.numRows()
    val m: Double = Math.floor(b*n)

    var eps: Double = Double.MaxValue


    var tk: Double = 1.0
    var tkm1: Double = tk
    var zqm1: BDM[Double] = BDM.zeros[Double](d.toInt, 1)
    var gam: Double = gamma //moving gamma

    // Initialize the weight parameters
    // Use this line to load in the default (or randomly initialized vector)
    // Use coordinate matrices for API uses
    val w0: BDM[Double] = BDM.zeros[Double](d.toInt, 1)
    var w: BDM[Double] = w0; // Set our weights (the ones that change, equal to w0)
    // w should have the same number of partitions as the original matrix? - Check with Saeed
    var wm1: BDM[Double] = w0; // weights used 1 iteration ago
    var tick, tock: Long = System.currentTimeMillis()


    // Main algorithm loops
    breakable {
      for (i <- 0 to t / k) {
        tick = System.currentTimeMillis()
        val (gEntries, rEntries) = mutil.randomMatrixSamples(xDataT, yData, k, b, m, sc)

        val gMat: Array[BDM[Double]] = new Array[BDM[Double]](gEntries.length)
        val rMat: Array[BDM[Double]] = new Array[BDM[Double]](gEntries.length)

        for (i <- 0 until gEntries.length) {
          gMat(i) = mutil.RDDToBreeze(gEntries(i), d.toInt, d.toInt)
          rMat(i) = mutil.RDDToBreeze(rEntries(i), d.toInt, 1)
        }
        tock = System.currentTimeMillis()
        mutil.printTime(tick, tock, "Loop 1")

        tick = System.currentTimeMillis()

        breakable {
          for (j <- 1 to k) {

            val hb = gMat(j - 1) / m // To do normal multiplications
            val rb = rMat(j - 1) / m

            val fgrad = (X: BDM[Double]) => (hb * X) - rb

            val z0: BDM[Double] = w
            var zq: BDM[Double] = z0
            var sarg: BDM[Double] = null

            for (q <- 1 to Q) {
              gam = gam * beta
              tk = (1 + Math.sqrt(1 + (4 * (tkm1 * tkm1)))) / 2
              var v: BDM[Double] = null
              if ((i * k) + j - 1 > 0) {
                v = zq + ((zq - zqm1) *:* ((tkm1 - 1) / tk))
              } else {
                v = zq
              }

              sarg = v - (gam * fgrad(v))
              zqm1 = zq
              zq = mutil.Svec(sarg, gam * lambda)

              tkm1 = tk
            }
            wm1 = w
            w = zq
            if (wOpt != null) {
              if (wOpt.cols != w.cols || wOpt.rows != w.rows) {
                throw new IllegalArgumentException(s"Woptimal which was provided did not have proper dimensions of ($w.rows x $w.cols")
              } else {
                eps = norm(w.toDenseVector - wOpt.toDenseVector) / norm(wOpt.toDenseVector)
              }
            }

            if (eps < exitThreshold) {
              break
            }


            val o: Double = breeze.linalg.norm((mutil.coordToBreeze(xDataT.toCoordinateMatrix()) * w).toDenseVector -
              mutil.coordToBreeze(yData.toCoordinateMatrix()).toDenseVector)
            val o1 = 1.0 / (2 * n) * scala.math.pow(o, 2)
            val o2 = lambda * breeze.linalg.sum(breeze.numerics.abs(w))
            val objFunc = o1 + o2
            println(s"norm:$objFunc")
          }
        }

        if (eps < exitThreshold) {
          break
        }

        tock = System.currentTimeMillis()
        mutil.printTime(tick, tock, "Loop 2")
        println(w)
      }
    }

    mutil.breezeMatrixToCoord(w, sc)
  }
}
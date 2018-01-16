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

/*
Team Alpha Omega
CA-SFISTA implementation with Spark
*/

object CA_SFISTA {

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
             alpha: Double = 0.01,
             lambda: Double = .1): CoordinateMatrix = {

    val entries = sc.parallelize(data) // create RDD of MatrixEntry of features
    val labelEntries = sc.parallelize(labels) // create RDD of MatrixEntry of labels
    // TODO: Check if pre-partitioning data is necessary so we don't have
    // TODO: repartition in the loop. (entries/labelEntries)
    var xData: CoordinateMatrix = new CoordinateMatrix(entries)
    var yData: CoordinateMatrix = new CoordinateMatrix(labelEntries)
    val d = xData.numRows()
    val n = xData.numCols()

    val m: Double = Math.floor(b*n)

    // Initialize the weight parameters
    // Use this line to load in the default (or randomly initialized vector)
    // Use coordinate matrices for API uses
    val w0: BDM[Double] = BDM.zeros[Double](d.toInt, 1)
    var w: BDM[Double] = w0; // Set our weights (the ones that change, equal to w0)
    // w should have the same number of partitions as the original matrix? - Check with Saeed
    var wm1: BDM[Double] = w0; // weights used 1 iteration ago
    var wm2: BDM[Double] = null; // weights used 2 iterations ago
    var tick, tock: Long = System.currentTimeMillis()

    // Main algorithm loops
    for (i <- 0 to t / k) {
      var gEntries: Array[RDD[MatrixEntry]] = new Array(k)//sc.emptyRDD[MatrixEntry] // RDD for the sample matrices
      var rEntries: Array[RDD[MatrixEntry]] = new Array(k)//sc.emptyRDD[MatrixEntry]

      // Use a array of RDD's instead

      tick = System.currentTimeMillis()
      for (j <- 1 to k) {

        gEntries(j-1) = sc.emptyRDD[MatrixEntry]
        rEntries(j-1) = sc.emptyRDD[MatrixEntry]
        // randomized sampling of data from the RDDs

        // This doesn't work because it doesn't select full rows/columns
        // var samples = entries.sample(false, b, 1L).collect()

        var samples: RDD[MatrixEntry] = mutil.sampleRows(xData.transpose(), b).map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
        var sampleRDD: RDD[MatrixEntry] = samples.repartition(numPartitions)

        // "Real" labeled data

        // This doesn't work because it doesn't select full rows/columns
        //var labelSamples = labelEntries.sample(false, b, 1L).collect()
        val labelSamples = mutil.sampleRows(yData, b)
        val labelSampleRDD = labelSamples.repartition(numPartitions)

        // create two temporary CoordinateMatrix
        val coordX: RDD[MatrixEntry] = sampleRDD // Coordinate matrix of sampled matrix
        val coordXT: RDD[MatrixEntry] = coordX.map({case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})// ?
        val coordY: RDD[MatrixEntry] = labelSampleRDD


        var tmp = mutil.RDDMult(coordX, coordXT) //Gj
        gEntries(j-1) = gEntries(j-1).union(tmp)

        tmp = mutil.RDDMult(coordX, coordY)
        rEntries(j-1) = rEntries(j-1).union(tmp)

        gEntries(j-1) = gEntries(j-1).map({ case MatrixEntry(x, y, z) => MatrixEntry(x, y, z/m)})
        rEntries(j-1) = rEntries(j-1).map({ case MatrixEntry(x, y, z) => MatrixEntry(x, y, z/m)})
      }


      // collect the data as a demonstration of action
//      gEntries.collect()
//      rEntries.collect()
      tock = System.currentTimeMillis()
      mutil.printTime(tick, tock, "Loop 1")

      tick = System.currentTimeMillis()
      for (j <- 1 to k) {

        val hb = mutil.coordToBreeze(new CoordinateMatrix(gEntries(j-1))) // To do normal multiplications
        val hw: BDM[Double] = hb * w // Matrix mult

        val rb = mutil.coordToBreeze(new CoordinateMatrix(rEntries(j-1)))
        var fgrad: BDM[Double] = hw -  rb // Subtracts r from hw // matrix op

        var ikj: Double = (i*k + j - 2).toDouble / (i*k + j)

        var v: BDM[Double] = w + (ikj * (w - wm1)) // Elementwise

        // Argument passed to svec
        val sarg: BDM[Double] = v - (alpha * fgrad)


        // Set weights for next iteration
        wm2 = wm1
        wm1 = w
        w = mutil.Svec(sarg, lambda*alpha)
        val o: Double = breeze.linalg.norm((mutil.coordToBreeze(xData).t * w).toDenseVector - mutil.coordToBreeze(yData).toDenseVector)
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
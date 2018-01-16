/** This modules contains an implementation of CA-SFISTA from the paper
  *
  * Avoiding Communication in Proximal Methods for Convex Optimization Problems
  * https://arxiv.org/abs/1710.08883
  *
  * We use Apache Spark to solve a distributed least-squares regression
  */

package org.paramath

import scala.io.Source
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD

import scala.util.Random

/*
Team Alpha Omega
CA-SFISTA implementation with Spark
*/

object CA_SPNM {

  /**
    * CA_SPNM Implementation.
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
        var samples: RDD[MatrixEntry] = sampleRows(xData.transpose(), b).map({ case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})
        var sampleRDD: RDD[MatrixEntry] = samples.repartition(numPartitions)

        // "Real" labeled data

        // This doesn't work because it doesn't select full rows/columns
        //var labelSamples = labelEntries.sample(false, b, 1L).collect()
        val labelSamples = sampleRows(yData, b)
        val labelSampleRDD = labelSamples.repartition(numPartitions)

        // create two temporary CoordinateMatrix
        val coordX: RDD[MatrixEntry] = sampleRDD // Coordinate matrix of sampled matrix
        val coordXT: RDD[MatrixEntry] = coordX.map({case MatrixEntry(i, j, k) => MatrixEntry(j, i, k)})// ?
        val coordY: RDD[MatrixEntry] = labelSampleRDD


        var tmp = RDDMult(coordX, coordXT) //Gj
        gEntries(j-1) = gEntries(j-1).union(tmp)

        tmp = RDDMult(coordX, coordY)
        rEntries(j-1) = rEntries(j-1).union(tmp)

        gEntries(j-1) = gEntries(j-1).map({ case MatrixEntry(x, y, z) => MatrixEntry(x, y, z/m)})
        rEntries(j-1) = rEntries(j-1).map({ case MatrixEntry(x, y, z) => MatrixEntry(x, y, z/m)})
      }


      // collect the data as a demonstration of action
      //      gEntries.collect()
      //      rEntries.collect()
      tock = System.currentTimeMillis()
      printTime(tick, tock, "Loop 1")

      tick = System.currentTimeMillis()
      for (j <- 1 to k) {

        val hb = coordToBreeze(new CoordinateMatrix(gEntries(j-1))) // To do normal multiplications
        val hw: BDM[Double] = hb * w // Matrix mult

        val rb = coordToBreeze(new CoordinateMatrix(rEntries(j-1)))
        var fgrad: BDM[Double] = hw -  rb // Subtracts r from hw // matrix op

        var ikj: Double = (i*k + j - 2).toDouble / (i*k + j)

        var v: BDM[Double] = w + (ikj * (w - wm1)) // Elementwise

        // Argument passed to svec
        val sarg: BDM[Double] = v - (alpha * fgrad)


        // Set weights for next iteration
        wm2 = wm1
        wm1 = w
        w = Svec(sarg, lambda*alpha)
        val o: Double = breeze.linalg.norm((coordToBreeze(xData).t * w).toDenseVector - coordToBreeze(yData).toDenseVector)
        val o1 = 1.0/(2*n) * scala.math.pow(o, 2)
        val o2 = lambda * breeze.linalg.sum(breeze.numerics.abs(w))
        val objFunc = o1 + o2

        println(s"norm:$objFunc")
      }
      tock = System.currentTimeMillis()
      printTime(tick, tock, "Loop 2")
      println(w)
    }

    breezeMatrixToCoord(w, sc)
  }

  def printTime(tick: Long, tock: Long, id: String) {
    val diff = tock-tick
    println(s"Code section $id took $diff milliseconds to run")
  }

  /**
    * Convert a coordinate matrix to a breeze matrix
    * @param data The coordinate matrix
    * @return The breeze matrix
    */
  def coordToBreeze(data: CoordinateMatrix, r: Int = -1, c: Int = -1): BDM[Double] = {
    val rows: Int = if (r == -1) data.numRows().toInt else r
    val cols: Int = if (c == -1) data.numCols().toInt else c
    var m: BDM[Double] = BDM.zeros(rows, cols)
    data.entries.collect().foreach(mel => {
      m(mel.i.toInt, mel.j.toInt) = mel.value
    })
    m
  }

  /**
    * Shifts the entries of a matrix left/right and up/down
    * @param A RDD of MatrixEntry
    * @param rowShift Amount of shift for rows (+/-)
    * @param colShift Amount of shift for columns (+/-)
    * @return RDD of shifted entries.
    */
  def shiftEntries(A: RDD[MatrixEntry], rowShift: Long, colShift: Long ): RDD[MatrixEntry] = {
    A.map({ case MatrixEntry(i, j, k) => MatrixEntry(i+rowShift, j+colShift, k ) })
  }

  /**
    * Take a percent of the columns of a matrix.
    * @param A matrix to take column samples of
    * @param percent percent of columns to take
    * @return RDD[MatrixEntry] with columns shifted to create a normal matrix
    */
  def sampleRows(A: CoordinateMatrix, percent: Double): RDD[MatrixEntry] = {
    val nrows = A.numRows()
    val pickNum = Math.floor(nrows*percent)
    if (pickNum < 1) {
      throw new IllegalArgumentException(s"Sample $percent is too low for matrix with $nrows columns")
    }

    val cols: Array[Int] = uniqueRandVals(0, nrows.toInt-1, pickNum.toInt) // Get the column indexes
    var colSet: Set[Int] = cols.toSet
    scala.util.Sorting.quickSort(cols) // Sort the picked column indexes

    // Now we need to pick out the columns.
    // First, transpose to rows
    // Filter out the rows we don't want
    // Set corresponding row index to their index in the array
    // Convert back to row matrix and transpose

    var filteredrows = A.toIndexedRowMatrix().rows
      .filter({ case IndexedRow(i, v) => colSet.contains(i.toInt) })
      .map({case IndexedRow(i, v) => IndexedRow(binSearch(cols, i.toInt).toLong, v)})

    new IndexedRowMatrix(filteredrows).toCoordinateMatrix().entries // Convert back to Coordinate matrix.
  }

  /**
    * Basic binary search - O(logn)
    * @param a The array
    * @param num Number to search for
    * @return Index of the number in the array
    */
  def binSearch(a: Array[Int], num: Int): Int = {
    var lo: Int = 0
    var hi: Int = a.length-1
    var mid: Int = (hi+lo) >>> 1
    while (lo <= hi) {
      mid = (hi+lo) >>> 1
      if (a(mid) == num) {
        return mid
      } else if (num < a(mid)){
        hi = mid - 1
      } else {
        lo = mid + 1
      }
    }
    -1
  }

  /**
    * Generates a sequence of unique random numbers in the range [min, max]
    * @param min The lowest number in the range
    * @param max The largest number in the range
    * @param n The number of samples to pull from the range.
    */
  def uniqueRandVals(min: Int, max: Int, n: Int): Array[Int] = {
    val rang = min to max
    val arr: Array[Int] = rang.toArray[Int]
    for (i <- (max-min) to 1 by -1) {
      var j: Int = Math.abs(Random.nextInt()) % i
      val tmp = arr(j)
      arr(j) = arr(i)
      arr(i) = tmp
    }

    arr.slice(0, n)
  }

  /**
    * Read the libSVM data file
    * @param fileLoc
    * @return (data, labels)
    */
  def readSVMData(fileLoc: String): (Seq[MatrixEntry], Seq[MatrixEntry]) = {
    // manually read data and create separate matrices for label and features
    var i = 0
    val filename = fileLoc
    var thisSeq = Seq[MatrixEntry]()
    var labelSeq = Seq[MatrixEntry]()

    for (line <- Source.fromFile(filename).getLines) {
      val items = line.split(' ')
      val label = items.head.toDouble
      val thisLabelEntry = MatrixEntry(i, 0, label)
      labelSeq = labelSeq :+ thisLabelEntry

      for (para <- items.tail) {
        val indexAndValue = para.split(':')
        val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
        val value = indexAndValue(1).toDouble
        val thisEntry = MatrixEntry(i, index, value)
        thisSeq = thisSeq :+ thisEntry
      }
      i = i + 1
    }
    (thisSeq, labelSeq)
  }


  /**
    * Convert a breeze matrix to a coordinate matrix
    * @param A The breeze Matrix
    * @param sc Spark Context (used to parallelize)
    * @return A new CoordinateMatrix
    */
  def breezeMatrixToCoord(A: BDM[Double], sc: SparkContext): CoordinateMatrix = {
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
    * Turns a breeze vector into a distributed coordinate matrix
    * @param A The vector to convert
    * @param sc The spark context
    * @return The coordinate matrix representing the vector
    */
  def breezeVectorToCoord(A: BDV[Double], sc: SparkContext): CoordinateMatrix = {
    var x: Matrix = Matrices.dense(A.length, 1, A.data)
    val cols = x.toArray.grouped(x.numRows)
    var rows: Seq[Seq[Double]] = cols.toSeq.transpose
    var irows = rows.zipWithIndex.map(({ case (r, i) => (i, new DenseVector(r.toArray)) }))
    val vecs = irows.map({ case (i, vec) => new IndexedRow(i, vec) })
    val rddtemp: RDD[IndexedRow] = sc.parallelize(vecs, 2)
    var a = new IndexedRowMatrix(rddtemp)
    a.toCoordinateMatrix()
  }


  /**
    * Multiply two Matrix Entry RDDs together.
    * @param rdd1
    * @param rdd2
    * @return The multiplication of the 2 RDD's
    */
  def RDDMult(rdd1: RDD[MatrixEntry], rdd2: RDD[MatrixEntry]): RDD[MatrixEntry] = {
    var a: RDD[(Long, (Long, Double))] = rdd1.map({ case MatrixEntry(i, j, k) => (j, (i, k)) })
    var b: RDD[(Long, (Long, Double))] = rdd2.map({ case MatrixEntry(i, j, k) => (i, (j, k)) })
    a.join(b)
      .map({ case (_, ((i, v), (k, w))) => ((i, k), (v * w)) })
      .reduceByKey(_ + _)
      .map({ case ((i, k), sum) => MatrixEntry(i, k, sum) })
  }

  def RDDAdd(rdd1: RDD[MatrixEntry], rdd2: RDD[MatrixEntry], sub: Boolean = false): RDD[MatrixEntry] = {
    var a: RDD[((Long, Long), Double)] = rdd1.map({ case MatrixEntry(i, j, k) => ((i, j), k) })
    var b: RDD[((Long, Long), Double)] = rdd2.map({ case MatrixEntry(i, j, k) => ((i, j), k) })

    if (sub){
      a.union(b).reduceByKey(_ - _).map({ case ((i, k), sum) => MatrixEntry(i, k, sum)})
    } else {
      a.union(b).reduceByKey(_ + _).map({ case ((i, k), sum) => MatrixEntry(i, k, sum)})
    }
  }

  /**
    * Calculate the piecewise S_lambda function
    * @param weights The initial weights
    * @param lambda The lambda value
    * @return The newly calculated S vector (as a CoordinateMatrix)
    */
  def Svec(weights: RDD[MatrixEntry], lambda: Double): CoordinateMatrix = {

    var wvec: RDD[MatrixEntry] = weights.map( w => {
      if (w.value > lambda) {
        MatrixEntry(w.i, w.j, w.value - lambda)
      } else if (w.value < -1*lambda) {
        MatrixEntry(w.i, w.j, 0)
      } else {
        MatrixEntry(w.i, w.j, w.value + lambda)
      }
    })
    new CoordinateMatrix(wvec)
  }

  /**
    * Calculates the Soft-Thresholding Operator
    * See Eq. (7) in paper (Referenced at the top of this file)
    * @param weights
    * @param lambda
    * @return
    */
  def Svec(weights: BDM[Double], lambda: Double): BDM[Double] = {

    weights.map(f => {
      if (f > lambda) {
        f - lambda
      } else if (f < -1*lambda) {
        f + lambda
      } else {
        0
      }
    })
  }

}
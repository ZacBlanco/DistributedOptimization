package org.paramath

import breeze.linalg.DenseMatrix
import org.apache.spark.sql.SparkSession
import org.paramath.util.MathUtils

import scala.collection.mutable
import scala.io.Source

object SparkJob {

  def main(args: Array[String]): Unit = {
    val usage =
      """
         -{option} {Description | default}
         No default means the argument is required.

         -m {Spark master URL | spark://ke:7077 }
         -an {App Name | DistributedOptimizationTest }
         -f { input file location | null }
         -b { percent of columns to pick | .2 }
         -k { number of inner iterations | 10 }
         -t { outer iterations * k | 100 }
         -Q { Number of SFISTA/SPNM Iterations | 10 }
         -g { gamma parameter | .01 }
         -l { lambda parameter | .1 }
         -wopt { optimal weight vector filename, line by line | null }

      """.stripMargin

    var m = mutable.Map[String, String]()
    m += ("-m" -> "spark://ke:7077")
    m += ("-an" -> "DistributedOptimizationTest")
    m += ("-f" -> null)
    m += ("-b" -> ".2")
    m += ("-k" -> "10")
    m += ("-t" -> "100")
    m += ("-Q" -> "10")
    m += ("-g" -> ".01")
    m += ("-l" -> ".1")
    m += ("-wopt" -> null)

    var i: Int= 0
    while (i < args.length) {
      if (m.contains(args(i))) {
        m -= args(i)
        m += args(i) -> args(i+1)
      } else {
        println(usage)
        sys.exit(1)
      }
      i += 2
    }

    if (m.get("-f").get == null) {
      println("File location is required")
      println(usage)
      sys.exit(1)
    }
    var wopt: DenseMatrix[Double] = null
    if (m.get("-wopt").get != null) {
      try{
        val f = Source.fromFile(m.get("-wopt").get)
        val ct = f.getLines().length
        f.close()
        var i = 0
        var w = DenseMatrix.zeros[Double](ct, 1)
        for (l <- Source.fromFile(m.get("-wopt").get).getLines) {
          w(i, 0) = l.stripMargin.toDouble
          i += 1
        }
        wopt = w
      } catch {
        case e: Exception => {
          println("Error while reading optimal weight file")
          println(e)
          println("Continuing execution without optimal weights known")
        }
      }
    }


    var spark = SparkSession.builder
      .master(m.get("-m").get)
      .appName(m.get("-an").get)
      .getOrCreate()
    var sc = spark.sparkContext


    var (data, labels) = MathUtils.sparkRead(m.get("-f").get, sc)
    var tick = System.currentTimeMillis()
    val kMax = 500

//    for (p <- 1 until kMax by 200) {
      LASSOSolver(sc, data, labels,
        b=m.get("-b").get.toDouble,
        k=m.get("-k").get.toInt,
        t=m.get("-t").get.toInt,
        Q=m.get("-Q").get.toInt,
        gamma=m.get("-g").get.toDouble,
        lambda=m.get("-l").get.toDouble,
        wOpt=wopt)
//    }


    var tock = System.currentTimeMillis()
    MathUtils.printTime(tick, tock, "Overall")
  }
}

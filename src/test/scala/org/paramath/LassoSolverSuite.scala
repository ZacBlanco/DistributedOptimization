package org.paramath

import java.io.{BufferedWriter, File, FileWriter}

import org.paramath.LASSOSolver
import org.paramath.util.MathUtils

import scala.io.Source

class LassoSolverSuite extends SparkTestSuite {

  test("Abalone") {
    val (data, labels) = MathUtils.sparkRead("abalone.txt", sc)
    LASSOSolver(sc, data, labels)
  }

  test("SparkJob abalone") {

var testf =
  """2.17400308515330
    0
    0
    0
    5.36052852801644
    0
    0
    0""".stripMargin
    val f = new File("optweight")
    val bw = new BufferedWriter(new FileWriter(f))
    bw.write(testf)
    bw.close()
    SparkJob.main(Array("-wopt", "optweight", "-f", "abalone.txt", "-l", "1"))
    f.delete()
  }


}



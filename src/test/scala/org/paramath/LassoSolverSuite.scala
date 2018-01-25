package org.paramath

import org.paramath.LASSOSolver
import org.paramath.util.MathUtils

class LassoSolverSuite extends SparkTestSuite {

  test("Abalone") {
    val (data, labels) = MathUtils.sparkRead("abalone.txt", sc)
    LASSOSolver(sc, data, labels)
  }


}



name := "CA_SFISTA"

version := "0.1"

scalaVersion := "2.12.4"

libraryDependencies ++= {
  val sparkVer = "2.2.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % "provided" withSources(),
    "org.apache.spark" %% "spark-mllib" % sparkVer % "provided" withSources(),
    "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

  )
}
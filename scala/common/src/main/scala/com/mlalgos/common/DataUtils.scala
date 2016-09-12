package com.mlalgos.common

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object DataUtils {
  val SPACE  = " "

  def read_train_data(fileName: String): INDArray = {
   Nd4j.create(
     Source.fromFile(fileName).getLines().map(
       _.stripLineEnd.split(SPACE, -1).map(_.toDouble)
     ).toList.toArray
   )
  }

  def read_test_data(fileName: String): INDArray = {
    Nd4j.create(
      Source.fromFile(fileName).getLines().map(
        _.stripLineEnd.split(SPACE, -1).map(_.toDouble)
      ).toList.toArray
    )
  }
}
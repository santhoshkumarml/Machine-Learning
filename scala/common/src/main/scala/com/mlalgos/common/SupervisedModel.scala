package com.mlalgos.common

import org.nd4j.linalg.api.ndarray.INDArray

trait SupervisedModel {
  /**
    * Train the model
    *
    * @param train_data training data
    * @param train_result training result
    */
  def fit(train_data: INDArray, train_result: INDArray) : Unit

  /**
    * Predict result for the dataSet
    * @param iNDArray feature vector
    */
  def predict(iNDArray: INDArray): Any
}

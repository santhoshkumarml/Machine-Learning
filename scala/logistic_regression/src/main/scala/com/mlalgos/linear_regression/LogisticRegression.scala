package com.mlalgos.linear_regression

import com.mlalgos.common.SupervisedModel
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

final class LogisticRegression(private var w : INDArray) extends SupervisedModel{
  /**
    * Train the model
    *
    * @param train_data training data
    * @param train_result training result
    */
  override def fit(train_data: INDArray, train_result: INDArray): Unit = {
    val (train_samples, no_of_features) = train_data.shape() match {case Array(a, b) => (a , b)}
    w = Nd4j.zeros(no_of_features + 1) //We take a transpose and do dot product w0 = 1 always
  }

  /**
    * Predict result for the dataSet
    *
    * @param iNDArray feature vector
    */
  override def predict(iNDArray: INDArray): Any = {

  }
}
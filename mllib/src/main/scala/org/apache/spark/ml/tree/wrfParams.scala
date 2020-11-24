/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree

import java.util.Locale

import scala.util.Try

import org.apache.spark.annotation.Since
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._

private[ml] trait WeightedRandomForestParams extends RandomForestParams {

  /**
   * Add featureWeight Params
   * @group param
   */
  final val featureWeight: DoubleArrayParam =
    new DoubleArrayParam(this, "featureWeight", "The probability of selecting each feature when selecting candidate features",
      _.forall(x => x >= 0))

  setDefault(featureWeight -> Array.fill[Double](0)(1.0))

  /** @group getParam */
  final def getFeatureWeight: Array[Double] = $(featureWeight)

  final val numIteration: IntParam =
    new IntParam(this, "numIteration", "The number of iterations for reweighted RF training", _ >= 1)

  setDefault(numIteration -> 1)
  /** @group getParam */
  final def getNumIteration: Int = $(numIteration)
    
    
  final val dataSubSamplingRate: DoubleArrayParam =
    new DoubleArrayParam(this, "dataSubSamplingRate", "The data sub-sampling rate is ", _.forall(x => x >= 0))

  setDefault(dataSubSamplingRate -> Array.fill[Double](0)(1.0))
  /** @group getParam */
  final def getDataSubSamplingRate: Array[Double] = $(dataSubSamplingRate)
   
   final val repopulate: BooleanParam =
    new BooleanParam(this, "repopulate", "If false, the " +
    " algorithm will use bagged input to populate leaves. If true, the  " + "algorithm will populate leaves with the entire training set.")
    
  setDefault(repopulate -> false)

     final def getRepopulate: Boolean = $(repopulate)
    
    
   final val useBinned: BooleanParam =
    new BooleanParam(this, "useBinned", "If false, the " +
    " algorithm will not use binned values to populate leaves. If true, the  " + "algorithm will use binned values.")
    
  setDefault(useBinned -> false)

     final def getUseBinned: Boolean = $(useBinned)
    
    final val useOOB:  BooleanParam =
    new BooleanParam(this,"useOOB", "If false, the " +
    " algorithm will not use OOB samples to populate leaves. If true, the  " + "algorithm will use OOB binned values.")
    
    setDefault(useOOB -> false)
    
    final def getUseOOB: Boolean = $(useOOB)
}

private[ml] trait WeightedRandomForestClassifierParams
  extends WeightedRandomForestParams with TreeEnsembleClassifierParams with TreeClassifierParams

private[ml] trait WeightedRandomForestRegressorParams
  extends WeightedRandomForestParams with TreeEnsembleRegressorParams with TreeRegressorParams


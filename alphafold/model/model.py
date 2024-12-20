# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union, List

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
from alphafold.model import modules_multimer
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree
import os
import pickle
import sys


def get_confidence_metrics(
    prediction_result: Mapping[str, Any], multimer_mode: bool
) -> Mapping[str, Any]:
    """Post processes prediction_result to get confidence metrics."""
    confidence_metrics = {}
    if "predicted_lddt" in prediction_result:
        confidence_metrics["plddt"] = confidence.compute_plddt(
            prediction_result["predicted_lddt"]["logits"]
        )
    if "predicted_aligned_error" in prediction_result:
        confidence_metrics.update(
            confidence.compute_predicted_aligned_error(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
            )
        )
        confidence_metrics["ptm"] = confidence.predicted_tm_score(
            logits=prediction_result["predicted_aligned_error"]["logits"],
            breaks=prediction_result["predicted_aligned_error"]["breaks"],
            asym_id=None,
        )
        if multimer_mode:
            # Compute the ipTM only for the multimer model.
            confidence_metrics["iptm"] = confidence.predicted_tm_score(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
                asym_id=prediction_result["predicted_aligned_error"]["asym_id"],
                interface=True,
            )
            # confidence_metrics["ranking_confidence"] = (
            #     0.8 * confidence_metrics["iptm"] + 0.2 * confidence_metrics["ptm"]
            # )

    # if not multimer_mode:
    #     # Monomer models use mean pLDDT for model ranking.
    #     confidence_metrics["ranking_confidence"] = np.mean(confidence_metrics["plddt"])

    return confidence_metrics

# Calculate ipTM/pTM from the intermediate pair representations
def get_extra_confidence_metrics(
    prediction_result: Mapping[str, Any],
    multimer_mode: bool,
    extra_output_layers: List[int],
) -> Mapping[str, Any]:
    """Post processes prediction_result to get confidence metrics."""
    extra_confidence_metrics = {}

    if len(extra_output_layers) == len(prediction_result["pae_layer_values"]):
        for i in range(len(extra_output_layers)):
            # pae_layer_name = f"predicted_aligned_error_layer_{i}"
            # if pae_layer_name in prediction_result:
            # extra_confidence_metrics.update(
            #     confidence.compute_extra_predicted_aligned_error(
            #         logits=prediction_result[pae_layer_name]["logits"],
            #         breaks=prediction_result[pae_layer_name]["breaks"],
            #         layer=i,
            #     )
            # )
            extra_confidence_metrics[f"ptm_layer_{extra_output_layers[i]}"] = confidence.predicted_tm_score(
                logits=prediction_result["pae_layer_values"][i]["logits"],
                breaks=prediction_result["pae_layer_values"][i]["breaks"],
                asym_id=None,
            )
            extra_confidence_metrics[f"logits_layer_{extra_output_layers[i]}"] = prediction_result["pae_layer_values"][i]["logits"]
            if multimer_mode:
                # Compute the ipTM only for the multimer model.
                
                extra_confidence_metrics[f"iptm_layer_{extra_output_layers[i]}"] = confidence.predicted_tm_score(
                        logits=prediction_result["pae_layer_values"][i]["logits"],
                        breaks=prediction_result["pae_layer_values"][i]["breaks"],
                        asym_id=prediction_result["asym_id"],
                        interface=True,
                    )
                
    return extra_confidence_metrics



class RunModel:
    """Container for JAX model."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        output_dir: str,
        params: Optional[Mapping[str, Mapping[str, jax.Array]]] = None,
    ):
        self.config = config
        self.params = params
        self.multimer_mode = config.model.global_config.multimer_mode

        if self.multimer_mode:

            def _forward_fn(batch):
                # logging.info("self.config.model: %s", str(self.config.model))
                model = modules_multimer.AlphaFold(self.config.model)
                result = model(batch, is_training=False)
                # this happens after return ret
                # this does happen
                # logging.info("###################### model(batch..) finished ###########################")
                # logging.info("Result: %s", result)
                # logging.info("Result['msa']: %s", result['msa'])
                # result_output_path = output_dir+"representations.npy"#os.path.join(output_dir, f'representations.npy')
                # with open(result_output_path, 'wb') as f:
                # pickle.dump(result, f, protocol=4)
                # jax.numpy.save(result_output_path, result['msa'])
                # sys.exit()
                return result

        else:

            def _forward_fn(batch):
                model = modules.AlphaFold(self.config.model)
                return model(
                    batch,
                    is_training=False,
                    compute_loss=False,
                    ensemble_representations=True,
                )

        self.apply = jax.jit(hk.transform(_forward_fn).apply)
        self.init = jax.jit(hk.transform(_forward_fn).init)

    def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
        """Initializes the model parameters.

        If none were provided when this class was instantiated then the parameters
        are randomly initialized.

        Args:
          feat: A dictionary of NumPy feature arrays as output by
            RunModel.process_features.
          random_seed: A random seed to use to initialize the parameters if none
            were set when this class was initialized.
        """
        if not self.params:
            # Init params randomly.
            rng = jax.random.PRNGKey(random_seed)
            self.params = hk.data_structures.to_mutable_dict(self.init(rng, feat))
            logging.warning("Initialized parameters randomly")

    def process_features(
        self,
        raw_features: Union[tf.train.Example, features.FeatureDict],
        random_seed: int,
    ) -> features.FeatureDict:
        """Processes features to prepare for feeding them into the model.

        Args:
          raw_features: The output of the data pipeline either as a dict of NumPy
            arrays or as a tf.train.Example.
          random_seed: The random seed to use when processing the features.

        Returns:
          A dict of NumPy feature arrays suitable for feeding into the model.
        """

        if self.multimer_mode:
            return raw_features

        # Single-chain mode.
        if isinstance(raw_features, dict):
            return features.np_example_to_features(
                np_example=raw_features, config=self.config, random_seed=random_seed
            )
        else:
            return features.tf_example_to_features(
                tf_example=raw_features, config=self.config, random_seed=random_seed
            )

    def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
        self.init_params(feat)
        logging.info(
            "Running eval_shape with shape(feat) = %s",
            tree.map_structure(lambda x: x.shape, feat),
        )
        shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
        logging.info("Output shape was %s", shape)
        return shape

    def predict(
        self,
        feat: features.FeatureDict,
        random_seed: int,
    ) -> Mapping[str, Any]:
        """Makes a prediction by inferencing the model on the provided features.

        Args:
          feat: A dictionary of NumPy feature arrays as output by
            RunModel.process_features.
          random_seed: The random seed to use when running the model. In the
            multimer model this controls the MSA sampling.

        Returns:
          A dictionary of model outputs.
        """
        # logging.info(
        #    "################### Init params #################################"
        # )
        self.init_params(feat)
        # logging.info(
        #    "Running predict with shape(feat) = %s",
        #    tree.map_structure(lambda x: x.shape, feat),
        # )
        # logging.info(
        #    "################### Self.apply start #################################"
        # )
        # logging.info("feat: %s", str(feat.keys()))
        # logging.info("self.params: %s", str(self.params))
        result = self.apply(
            self.params, jax.random.PRNGKey(random_seed), feat
        )  # this is where it goes wrong
        # logging.info(
        #    "################### Self.apply finished #################################"
        # )
        # This block is to ensure benchmark timings are accurate. Some blocking is
        # already happening when computing get_confidence_metrics, and this ensures
        # all outputs are blocked on.
        jax.tree_map(lambda x: x.block_until_ready(), result)
        # logging.info(
        #    "################### jax.tree_map finished #################################"
        # )
        # logging.info(
        #    "Output shape was %s", tree.map_structure(lambda x: x.shape, result)
        # )
        #confidence_metrics = get_confidence_metrics(result, multimer_mode=self.multimer_mode)
        #result["iptm"] = confidence_metrics["iptm"]
        result.update(
            get_confidence_metrics(result, multimer_mode=self.multimer_mode))

        result.update(
            get_extra_confidence_metrics(
                result,
                multimer_mode=self.multimer_mode,
                extra_output_layers=self.config.model.embeddings_and_evoformer.extra_evoformer_output_layers,
            )
        )
        if 'pae_layer_values' in result.keys():
            del result['pae_layer_values']
        if 'intermediate_pair' in result.keys():
           del result['intermediate_pair']
        del result['msa']
        del result['aligned_confidence_probs']
        del result['predicted_aligned_error']
        del result['pair']
        del result["single"]
        del result["logits"]
        if 'structure_module' in result.keys():
            del result['structure_module']
        if 'distogram' in result.keys():
            del result['distogram']
        return result

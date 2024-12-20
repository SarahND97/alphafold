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

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Union

from absl import app
from absl import flags
from absl import logging
# from alphafold.common import confidence
# from alphafold.common import protein
# from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import folddock_pipeline
# from alphafold.data import templates
# from alphafold.data.tools import hhsearch
# from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model

# from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


flags.DEFINE_list(
    "fasta_paths",
    None,
    "Paths to FASTA files, each containing a prediction "
    "target that will be folded one after another. If a FASTA file contains "
    "multiple sequences, then it will be folded as a multimer. Paths should be "
    "separated by commas. All FASTA paths must have a unique basename as the "
    "basename is used to name the output directories for each prediction.",
)

flags.DEFINE_string("data_dir", None, "Path to directory of supporting data.")
flags.DEFINE_string(
    "output_dir", None, "Path to a directory that will " "store the results."
)
flags.DEFINE_boolean(
    "separate_output_dir", True, "Save the results separately according to protein name"
)
flags.DEFINE_string(
    "msa_dir",
    None,
    "Path to a directory that stores precomputed msas,"
    "if None the msas are assumed to be stored in the output_dir.",
)
flags.DEFINE_string(
    "model_to_run",
    "all",
    "Choose a specific model that you want to run"
    "choose between 1-5,all and random.",
)
flags.DEFINE_string(
    "jackhmmer_binary_path",
    shutil.which("jackhmmer"),
    "Path to the JackHMMER executable.",
)
flags.DEFINE_string(
    "hhblits_binary_path", shutil.which("hhblits"), "Path to the HHblits executable."
)
flags.DEFINE_string(
    "hhsearch_binary_path", shutil.which("hhsearch"), "Path to the HHsearch executable."
)
flags.DEFINE_string(
    "hmmsearch_binary_path",
    shutil.which("hmmsearch"),
    "Path to the hmmsearch executable.",
)
flags.DEFINE_string(
    "hmmbuild_binary_path", shutil.which("hmmbuild"), "Path to the hmmbuild executable."
)
flags.DEFINE_string(
    "kalign_binary_path", shutil.which("kalign"), "Path to the Kalign executable."
)
flags.DEFINE_string(
    "uniref90_database_path",
    None,
    "Path to the Uniref90 " "database for use by JackHMMER.",
)
flags.DEFINE_string(
    "mgnify_database_path", None, "Path to the MGnify " "database for use by JackHMMER."
)
flags.DEFINE_string(
    "bfd_database_path", None, "Path to the BFD " "database for use by HHblits."
)
flags.DEFINE_string(
    "small_bfd_database_path",
    None,
    "Path to the small " 'version of BFD used with the "reduced_dbs" preset.',
)
flags.DEFINE_string(
    "uniref30_database_path",
    None,
    "Path to the UniRef30 " "database for use by HHblits.",
)
flags.DEFINE_string(
    "uniprot_database_path",
    None,
    "Path to the Uniprot " "database for use by JackHMMer.",
)
flags.DEFINE_string(
    "pdb70_database_path", None, "Path to the PDB70 " "database for use by HHsearch."
)
flags.DEFINE_string(
    "pdb_seqres_database_path",
    None,
    "Path to the PDB " "seqres database for use by hmmsearch.",
)
flags.DEFINE_string(
    "template_mmcif_dir",
    None,
    "Path to a directory with " "template mmCIF structures, each named <pdb_id>.cif",
)
flags.DEFINE_string(
    "max_template_date",
    None,
    "Maximum template release date "
    "to consider. Important if folding historical test sets.",
)
flags.DEFINE_string(
    "obsolete_pdbs_path",
    None,
    "Path to file containing a "
    "mapping from obsolete PDB IDs to the PDB IDs of their "
    "replacements.",
)
flags.DEFINE_enum(
    "db_preset",
    "full_dbs",
    ["full_dbs", "reduced_dbs"],
    "Choose preset MSA database configuration - "
    "smaller genetic database config (reduced_dbs) or "
    "full genetic database config  (full_dbs)",
)
flags.DEFINE_enum(
    "model_preset",
    "monomer",
    ["monomer", "monomer_casp14", "monomer_ptm", "multimer"],
    "Choose preset model configuration - the monomer model, "
    "the monomer model with extra ensembling, monomer model with "
    "pTM head, or multimer model",
)
flags.DEFINE_boolean(
    "benchmark",
    False,
    "Run multiple JAX model evaluations "
    "to obtain a timing that excludes the compilation time, "
    "which should be more indicative of the time required for "
    "inferencing many proteins.",
)
flags.DEFINE_integer(
    "random_seed",
    None,
    "The random seed for the data "
    "pipeline. By default, this is randomly generated. Note "
    "that even if this is set, Alphafold may still not be "
    "deterministic, because processes like GPU inference are "
    "nondeterministic.",
)
flags.DEFINE_integer(
    "num_multimer_predictions_per_model",
    5,
    "How many "
    "predictions (each with a different random seed) will be "
    "generated per model. E.g. if this is 2 and there are 5 "
    "models then there will be 10 predictions per input. "
    "Note: this FLAG only applies if model_preset=multimer",
)
# parser.add_argument("-c", "--conditional-option", action="store_true", help="Conditional Option")
# parser.add_argument("-d", "--dependent-argument", required=False, help="Dependent Argument")

flags.DEFINE_boolean(
    "use_precomputed_msas",
    True,
    "Whether to read MSAs that "
    "have been written to disk instead of running the MSA "
    "tools. The MSA files are looked up in the output "
    "directory, so it must stay the same between multiple "
    "runs that are to reuse the MSAs. WARNING: This will not "
    "check if the sequence, database or configuration have "
    "changed.",
)
flags.DEFINE_enum_class(
    "models_to_relax",
    ModelsToRelax.BEST,
    ModelsToRelax,
    "The models to run the final relaxation step on. "
    "If `all`, all models are relaxed, which may be time "
    "consuming. If `best`, only the most confident model "
    "is relaxed. If `none`, relaxation is not run. Turning "
    "off relaxation might result in predictions with "
    "distracting stereochemical violations but might help "
    "in case you are having issues with the relaxation "
    "stage.",
)
flags.DEFINE_boolean(
    "use_gpu_relax",
    None,
    "Whether to relax on GPU. "
    "Relax on GPU can be much faster than CPU, so it is "
    "recommended to enable if possible. GPUs must be available"
    " if this setting is enabled.",
)
flags.DEFINE_string(
    "layers_to_calculate_iptm",
    "all",
    "Specify at which layers to calculate ipTM, choose between all, none or every X layer as X",
)

flags.DEFINE_boolean(
    "run_only_pae_head",
    False,
    "If True, run only PredictedAlignedErrorHead (in particular we skip the structure module)",
)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def _check_flag(flag_name: str, other_flag_name: str, should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = "be" if should_be_set else "not be"
        raise ValueError(
            f"{flag_name} must {verb} set when running with "
            f'"--{other_flag_name}={FLAGS[other_flag_name].value}".'
        )


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output

def _model_config(model_name, num_predictions_per_model,run_multimer_system,num_ensemble, data_dir, output_dir):
    model_runners = {}
    model_config = config.model_config(model_name)
    if run_multimer_system:
        model_config.model.num_ensemble_eval = num_ensemble
    else:
        model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=data_dir
    )
    model_config.model.num_recycle = 0  # want zero recycles
    model_runner = model.RunModel(
        config=model_config, params=model_params, output_dir=output_dir
    )
    for i in range(num_predictions_per_model):
        model_runners[f"{model_name}_pred_{i}"] = model_runner
    return model_runners
    
def predict_structure_modified(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline, pipeline.ModifiedDataPipeline, pipeline_multimer.ModifiedDataPipeline, folddock_pipeline.FoldDockPipeline],
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    msa_dir: str = None,
    output_dir: str = None,
    separate_output_dir: bool = True,
    paired_msa: str = None,
):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info("Predicting %s", fasta_name)
    timings = {}
    # Introduce the option of saving all representations in the same dir
    output_dir = ""
    if separate_output_dir:
        output_dir = os.path.join(output_dir_base, fasta_name)
    else:
        output_dir = output_dir_base
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, "msas")
    if not os.path.exists(msa_output_dir) and msa_dir is not None:
        os.makedirs(msa_output_dir)

    # make it possible to have msas in a different place than
    # where the results will be put
    if msa_dir:
        msa_output_dir = msa_dir

    # Get features.
    t_0 = time.time()
    # original pipeline
    # feature_dict = data_pipeline.process(
    #     input_fasta_path=fasta_path, paired_msa=paired_msa
    # )
    # folddock pipeline
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path, input_msas=[paired_msa]
    )
    timings["features"] = time.time() - t_0

    # Run the models.
    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        logging.info("Running model %s on %s", model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed
        )

        timings[f"process_features_{model_name}"] = time.time() - t_0

        t_0 = time.time()
        
        prediction_result = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed
        )
        
        t_diff = time.time() - t_0
        timings[f"predict_and_compile_{model_name}"] = t_diff
        logging.info(
            "Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs",
            model_name,
            fasta_name,
            t_diff,
        )
        np_prediction_result = _jnp_to_np(dict(prediction_result))
        print("np_prediction_result", np_prediction_result.keys())
        print('iptm', np_prediction_result["iptm"])
        return np_prediction_result["pair_2"], np_prediction_result["logits_layer_-1"] 

# function to get the pair and logits for the TMHead
def get_info_for_tmhead(model_preset, layers_to_calculate_iptm, fasta_paths, msa_dir, output_dir, data_dir, model_to_run, num_ensemble, paired_msa, random_seed_input=None): 

    run_multimer_system = "multimer" in model_preset
    
    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError("All FASTA paths must have a unique basename.")

    monomer_data_pipeline = pipeline.ModifiedDataPipeline(
        use_precomputed_msas=True,
    )

    if run_multimer_system:
        num_predictions_per_model = 1
        data_pipeline = pipeline_multimer.ModifiedDataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            use_precomputed_msas=True,
        )
    else:
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    data_pipeline = folddock_pipeline.FoldDockPipeline()

    # Update config so that it computes the iptm at the specified layers
    iptm_layers = layers_to_calculate_iptm
    upper_range = config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block + 1
    if iptm_layers == "all":
        iptm_layers = list(range(upper_range))
    elif iptm_layers == "none" or iptm_layers == "None" or iptm_layers is None:
        iptm_layers = []
    elif "," in iptm_layers:
        iptm_layers_temp = iptm_layers.split(",")
        iptm_layers = [int(layer) for layer in iptm_layers_temp]
    else:
        iptm_layers = list(range(0,upper_range,int(iptm_layers)))

    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.extra_evoformer_output_layers = (
        iptm_layers
    )
    
    model_runners = {}
    model_names = config.MODEL_PRESETS[model_preset]
    
    # add here the ability to pick out one model
    if model_to_run=="all":
        for model_name in model_names:
            model_runners = _model_config(model_name, num_predictions_per_model, run_multimer_system, num_ensemble, data_dir, output_dir)
    elif model_to_run=="random":
        model_runners = _model_config(random.choice(model_names), num_predictions_per_model, run_multimer_system, num_ensemble, data_dir, output_dir)
    else:
        model_int = int(model_to_run)
        model_runners = _model_config(model_names[model_int], num_predictions_per_model, run_multimer_system, num_ensemble, data_dir, output_dir)

    logging.info("Have %d models: %s", len(model_runners), list(model_runners.keys()))

    random_seed = random_seed_input
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info("Using random seed %d for the data pipeline", random_seed)

    # Predict structure for each of the sequences.
    for i, fasta_path in enumerate(fasta_paths):
        fasta_name = fasta_names[i]
        pair, logits = predict_structure_modified(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=output_dir,
            data_pipeline=data_pipeline,
            model_runners=model_runners,
            random_seed=random_seed,
            msa_dir=msa_dir,
            paired_msa=paired_msa,
        )

    return pair, logits

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    for tool_name in (
        "jackhmmer",
        "hhblits",
        "hhsearch",
        "hmmsearch",
        "hmmbuild",
        "kalign",
    ):
        if not FLAGS[f"{tool_name}_binary_path"].value:
            raise ValueError(
                f'Could not find path to the "{tool_name}" binary. Make '
                "sure it is installed on your system."
            )

    use_small_bfd = FLAGS.db_preset == "reduced_dbs"
    if not FLAGS.use_precomputed_msas:
        _check_flag("small_bfd_database_path", "db_preset", should_be_set=use_small_bfd)
        _check_flag("bfd_database_path", "db_preset", should_be_set=not use_small_bfd)
        _check_flag(
            "uniref30_database_path", "db_preset", should_be_set=not use_small_bfd
        )

    # If the msas are not pre-computed then we need the paths to the databases
    _check_flag(
        "uniref90_database_path",
        "not use_precomputed_msas",
        should_be_set=not FLAGS.use_precomputed_msas,
    )
    _check_flag(
        "mgnify_database_path",
        "db_preset",
        should_be_set=not FLAGS.use_precomputed_msas,
    )
    _check_flag(
        "max_template_date",
        "not use_precomputed_msas",
        should_be_set=not FLAGS.use_precomputed_msas,
    )
    _check_flag(
        "max_template_date",
        "not use_precomputed_msas",
        should_be_set=not FLAGS.use_precomputed_msas,
    )

    run_multimer_system = "multimer" in FLAGS.model_preset
    model_type = "Multimer" if run_multimer_system else "Monomer"
    if not FLAGS.use_precomputed_msas:
        _check_flag(
            "pdb70_database_path", "model_preset", should_be_set=not run_multimer_system
        )
        _check_flag(
            "pdb_seqres_database_path",
            "model_preset",
            should_be_set=run_multimer_system,
        )
        _check_flag(
            "uniprot_database_path", "model_preset", should_be_set=run_multimer_system
        )

    if FLAGS.model_preset == "monomer_casp14":
        num_ensemble = 8
    else:
        num_ensemble = 1

    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError("All FASTA paths must have a unique basename.")

    ##### TEMPLATES #######
    template_searcher = None
    template_featurizer = None
    # if run_multimer_system:
    #   template_searcher = hmmsearch.Hmmsearch(
    #       binary_path=FLAGS.hmmsearch_binary_path,
    #       hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
    #       database_path=FLAGS.pdb_seqres_database_path)
    #   template_featurizer = templates.HmmsearchHitFeaturizer(
    #       mmcif_dir=FLAGS.template_mmcif_dir,
    #       max_template_date=FLAGS.max_template_date,
    #       max_hits=MAX_TEMPLATE_HITS,
    #       kalign_binary_path=FLAGS.kalign_binary_path,
    #       release_dates_path=None,
    #       obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
    # else:
    #   template_searcher = hhsearch.HHSearch(
    #       binary_path=FLAGS.hhsearch_binary_path,
    #       databases=[FLAGS.pdb70_database_path])
    #   template_featurizer = templates.HhsearchHitFeaturizer(
    #       mmcif_dir=FLAGS.template_mmcif_dir,
    #       max_template_date=FLAGS.max_template_date,
    #       max_hits=MAX_TEMPLATE_HITS,
    #       kalign_binary_path=FLAGS.kalign_binary_path,
    #       release_dates_path=None,
    #       obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
    )

    if run_multimer_system:
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
            uniprot_database_path=FLAGS.uniprot_database_path,
            use_precomputed_msas=FLAGS.use_precomputed_msas,
        )
    else:
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    
    # Update config so that it computes the iptm at the specified layers
    iptm_layers = FLAGS.layers_to_calculate_iptm
    upper_range = config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block + 1
    if iptm_layers == "all":
        iptm_layers = list(range(upper_range))
    elif iptm_layers == "none" or iptm_layers == "None" or iptm_layers is None:
        iptm_layers = []
    elif "," in iptm_layers:
        iptm_layers_temp = iptm_layers.split(",")
        iptm_layers = [int(layer) for layer in iptm_layers_temp]
    else:
        iptm_layers = list(range(0,upper_range,int(iptm_layers)))
    # print("######### iptm_layers ###################", iptm_layers)

    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.extra_evoformer_output_layers = (
        iptm_layers
    )
    
    # Uodate config to include run_only_pae_head
    config.CONFIG_MULTIMER.model.run_only_pae_head = FLAGS.run_only_pae_head

    model_runners = {}
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
    
    # add here the ability to pick out one model
    if FLAGS.model_to_run=="all":
        for model_name in model_names:
            model_runners = _model_config(model_name, num_predictions_per_model, run_multimer_system, num_ensemble)
    elif FLAGS.model_to_run=="random":
        model_runners = _model_config(random.choice(model_names), num_predictions_per_model, run_multimer_system, num_ensemble)
    else:
        model_int = int(FLAGS.model_to_run)
        model_runners = _model_config(model_names[model_int], num_predictions_per_model, run_multimer_system, num_ensemble)

    logging.info("Have %d models: %s", len(model_runners), list(model_runners.keys()))

    # amber_relaxer = relax.AmberRelaxation(
    #     max_iterations=RELAX_MAX_ITERATIONS,
    #     tolerance=RELAX_ENERGY_TOLERANCE,
    #     stiffness=RELAX_STIFFNESS,
    #     exclude_residues=RELAX_EXCLUDE_RESIDUES,
    #     max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    #     use_gpu=FLAGS.use_gpu_relax)

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info("Using random seed %d for the data pipeline", random_seed)

    # Predict structure for each of the sequences.
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        fasta_name = fasta_names[i]
        _, _ = predict_structure_modified(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=FLAGS.output_dir,
            data_pipeline=data_pipeline,
            model_runners=model_runners,
            amber_relaxer="",
            benchmark=FLAGS.benchmark,
            run_only_pae_head=FLAGS.run_only_pae_head,
            random_seed=random_seed,
            models_to_relax=FLAGS.models_to_relax,
            model_type=model_type,
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "output_dir",
            "data_dir",
            # "uniref90_database_path",
            # "mgnify_database_path",
            # "template_mmcif_dir",
            # "max_template_date",
            # "obsolete_pdbs_path",
            "use_gpu_relax",
        ]
    )

    app.run(main)
    
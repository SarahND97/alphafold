import os, sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
# from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
# from alphafold.data.tools import hhblits
# from alphafold.data.tools import hhsearch
# from alphafold.data.tools import hmmsearch
# from alphafold.data.tools import jackhmmer
# from alphafold.data import msa_pairing
from alphafold.data.pipeline import make_msa_features, make_sequence_features, FeatureDict
from alphafold.data.pipeline_multimer import convert_monomer_features
from alphafold.data.feature_processing import _crop_single_chain, process_final
from alphafold.data.msa_pairing import block_diag, SEQ_FEATURES
import numpy as np
# import scipy.linalg

# FeatureDict = MutableMapping[str, np.ndarray]
# TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

# def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
#     """Constructs a feature dict of MSA features."""
#     if not msas:
#         raise ValueError("At least one MSA must be provided.")

#     int_msa = []
#     deletion_matrix = []
#     species_ids = []
#     seen_sequences = set()
#     for msa_index, msa in enumerate(msas):
#         if not msa:
#             raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
#         for sequence_index, sequence in enumerate(msa.sequences):
#             if sequence in seen_sequences:
#                 continue
#             seen_sequences.add(sequence)
#             int_msa.append(
#                 [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
#             )
#             deletion_matrix.append(msa.deletion_matrix[sequence_index])
#             identifiers = msa_identifiers.get_identifiers(
#                 msa.descriptions[sequence_index]
#             )
#             species_ids.append(identifiers.species_id.encode("utf-8"))

#     num_res = len(msas[0].sequences[0])
#     num_alignments = len(int_msa)
#     features = {}
#     features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
#     features["msa"] = np.array(int_msa, dtype=np.int32)
#     features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
#     features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)
#     return features

# def block_diag(*arrs: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
#   """Like scipy.linalg.block_diag but with an optional padding value."""
#   ones_arrs = [np.ones_like(x) for x in arrs]
#   off_diag_mask = 1.0 - scipy.linalg.block_diag(*ones_arrs)
#   diag = scipy.linalg.block_diag(*arrs)
#   diag += (off_diag_mask * pad_value).astype(diag.dtype)
#   return diag

# def _crop_single_chain(chain: FeatureDict,
#                        msa_crop_size: int,
#                        pair_msa_sequences: bool,
#                        max_templates: int) -> FeatureDict:
#   """Crops msa sequences to `msa_crop_size`."""
#   msa_size = chain['num_alignments']

#   if pair_msa_sequences:
#     msa_size_all_seq = chain['num_alignments_all_seq']
#     msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)

#     # We reduce the number of un-paired sequences, by the number of times a
#     # sequence from this chain's MSA is included in the paired MSA.  This keeps
#     # the MSA size for each chain roughly constant.
#     msa_all_seq = chain['msa_all_seq'][:msa_crop_size_all_seq, :]
#     num_non_gapped_pairs = np.sum(
#         np.any(msa_all_seq != msa_pairing.MSA_GAP_IDX, axis=1))
#     num_non_gapped_pairs = np.minimum(num_non_gapped_pairs,
#                                       msa_crop_size_all_seq)

#     # Restrict the unpaired crop size so that paired+unpaired sequences do not
#     # exceed msa_seqs_per_chain for each chain.
#     max_msa_crop_size = np.maximum(msa_crop_size - num_non_gapped_pairs, 0)
#     msa_crop_size = np.minimum(msa_size, max_msa_crop_size)
#   else:
#     msa_crop_size = np.minimum(msa_size, msa_crop_size)

#   include_templates = 'template_aatype' in chain and max_templates
#   if include_templates:
#     num_templates = chain['template_aatype'].shape[0]
#     templates_crop_size = np.minimum(num_templates, max_templates)

#   for k in chain:
#     k_split = k.split('_all_seq')[0]
#     if k_split in msa_pairing.TEMPLATE_FEATURES and len(chain[k].shape)>1:
#       chain[k] = chain[k][:templates_crop_size, :]
#     elif k_split in msa_pairing.MSA_FEATURES:
#       if '_all_seq' in k and pair_msa_sequences:
#         chain[k] = chain[k][:msa_crop_size_all_seq, :]
#       else:
#         chain[k] = chain[k][:msa_crop_size, :]

#   chain['num_alignments'] = np.asarray(msa_crop_size, dtype=np.int32)
#   if include_templates:
#     chain['num_templates'] = np.asarray(templates_crop_size, dtype=np.int32)
#   if pair_msa_sequences:
#     chain['num_alignments_all_seq'] = np.asarray(
#         msa_crop_size_all_seq, dtype=np.int32)
#   return chain

# def make_sequence_features(
#     sequence: str, description: str, num_res: int
# ) -> FeatureDict:
#     """Constructs a feature dict of sequence features."""
#     features = {}
#     features["aatype"] = residue_constants.sequence_to_onehot(
#         sequence=sequence,
#         mapping=residue_constants.restype_order_with_x,
#         map_unknown_to_x=True,
#     )
#     features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
#     features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
#     features["residue_index"] = np.array(range(num_res), dtype=np.int32)
#     features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
#     features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
#     return features

class FoldDockPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self):
    pass
  
  def process_str(
        self,
        input_sequence,
        input_description = None,
  ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file""" 
        num_res = len(input_sequence)
        sequence_features = make_sequence_features(
          sequence=input_sequence,
          description=input_description,
          num_res=num_res,
          )
        return sequence_features
  
  def process_single_chain_sequence_features():
    pass

  def process(self, input_fasta_path: str, input_msas: list) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    
    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
        if len(input_seqs) > 2:
            raise ValueError(
                f"More than two input sequences found in {input_fasta_path}."
            )

    # Create sequence features    

    # all_sequence_features = []
    sequence_features = {} # 'aatype', 'between_segment_residues', 'domain_name', 'residue_index', 'seq_length', 'sequence'
    chain_id = ['A', 'B']
    for i, (input_sequence, input_description) in enumerate(zip(input_seqs, input_descs)):
        num_res = len(input_sequence)
        sequence_features_chain = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res
        )
        sequence_features_chain = convert_monomer_features(sequence_features_chain, chain_id[i])
        for key in sequence_features_chain.keys():
            if key not in sequence_features:
                sequence_features[key] = sequence_features_chain[key]
            # elif key in ['sequence','domain_name']:
            # #    print(f"{key}: ", sequence_features_chain[key])
            #    sequence_features[key] = sequence_features[key]+sequence_features_chain[key]
            #    sequence_features[key] = sequence_features[key][0]
            # elif key in ['aatype']:
            #     sequence_features[key] = np.concatenate((sequence_features[key], sequence_features_chain[key]), axis=0)

            # elif key in sequence_features:
            # ignoring auth_chain_id for now 
            elif key not in ['auth_chain_id', 'domain_name', 'seq_length', 'sequence']:
                print("key: ", key)
                sequence_features[key] = np.concatenate([sequence_features[key], sequence_features_chain[key]], axis=0)
            elif key in ['seq_length', 'sequence']: 
                sequence_features[key] = sequence_features[key] + sequence_features_chain[key]
            # else:
            #     sequence_features[key] = sequence_features_chain[key]
        # print("sequence_features.keys(): ", sequence_features.keys())
    # for feature_name, feats in sequence_features.items():
    #     if feature_name in  ['auth_chain_id', 'domain_name', 'seq_length', 'sequence']:
    #        print(f"feature_name: {feature_name} feature: {feats}")
    #        continue
    #     print(f"feature_name: {feature_name}, shape: {feats.shape}, type: {type(feats)}")
    # sys.exit()
    
    # for feature_name in sequence_features.keys():
    #     # feats = [x[feature_name] for x in sequence_features]
    #     # feature_name_split = feature_name.split('_all_seq')[0]
    #     print("feature_name: ", feature_name)
    #     print("feats: ", sequence_features[feature_name].shape)
        
                # all_sequence_features[str(i)] = {}
        # all_sequence_features[str(i)][input_description] = sequence_features
        # all_sequence_features += [sequence_features]
        
    num_res_1 = len(input_seqs[0])
    num_res_2 = len(input_seqs[1])

    # Extend existing values by concatenating with new arrays
    sequence_features["asym_id"] = np.concatenate(
        [sequence_features.get("asym_id", np.array([])), 1 * np.ones(num_res_1), 2 * np.ones(num_res_2)]
    )
    sequence_features["sym_id"] = np.concatenate(
        [sequence_features.get("sym_id", np.array([])), 1 * np.ones(num_res_1), 2 * np.ones(num_res_2)]
    )
    sequence_features["entity_id"] = np.concatenate(
        [sequence_features.get("entity_id", np.array([])), 1 * np.ones(num_res_1), 2 * np.ones(num_res_2)]
    )

    # Check shapes to confirm
    # print("sequence_features['asym_id'].shape:", sequence_features['asym_id'].shape)
    # print("sequence_features['sym_id'].shape:", sequence_features['sym_id'].shape)
    # print("sequence_features['entity_id'].shape:", sequence_features['entity_id'].shape)
    # for feature_name in sequence_features.keys():
    #     # feats = [x[feature_name] for x in sequence_features]
    #     # feature_name_split = feature_name.split('_all_seq')[0]
    #     print("feature_name: ", feature_name)
    #     print("feats: ", sequence_features[feature_name].shape)

    # print("sequence_features[sequence]: ", sequence_features['sequence'])
    # print("sequence_features['aatype'].shape: ", sequence_features['aatype'].shape)
    # sys.exit()

    # merge sequence features
    # merged_sequence_features = {}
    # for feature_name in all_sequence_features[0]:
    #     feats = [x[feature_name] for x in all_sequence_features]
    #     feature_name_split = feature_name.split('_all_seq')[0]
    #     if feature_name_split in SEQ_FEATURES:
    #         print("feature_name: ", feature_name)
    #         print("feats: ", np.array([feats[0]]).shape)
    #         merged_sequence_features[feature_name] = np.concatenate(feats, axis=0)
    # print("merged_sequence_features['aatype'].shape: ", merged_sequence_features['aatype'].shape)
    
    # Create msa features
    # parsed_msas = []
    # parsed_delmat = []
    msa_objects = []
    for custom_msa in input_msas:
      msa = ''.join([line for line in open(custom_msa)])
      if custom_msa[-3:] == 'sto':
        msa_object = parsers.parse_stockholm(msa)
        msa_objects.append(msa_object)
        # parsed_msa = msa_object.sequences
        # parsed_deletion_matrix = msa_object.deletion_matrix
      elif custom_msa[-3:] == 'a3m':
        msa_object = parsers.parse_a3m(msa)
        msa_objects.append(msa_object)
        # parsed_msa = msa_object.sequences
        # parsed_deletion_matrix = msa_object.deletion_matrix
      else: raise TypeError('Unknown format for input MSA, please make sure '
                            'the MSA files you provide terminates with (and '
                            'are formatted as) .sto or .a3m')
    #   parsed_msas.append(parsed_msa)
    #   parsed_delmat.append(parsed_deletion_matrix)
    
    msa_features = make_msa_features(msas=msa_objects)
         #, deletion_matrices=parsed_delmat)

    

    num_chains = 2
    # Convert deletion matrices to float.
    msa_features['deletion_matrix'] = np.asarray(msa_features.pop('deletion_matrix_int'), dtype=np.float32)
    if 'deletion_matrix_int_all_seq' in msa_features:
        msa_features['deletion_matrix_all_seq'] = np.asarray(
            msa_features.pop('deletion_matrix_int_all_seq'), dtype=np.float32)

    msa_features['deletion_mean'] = np.mean(msa_features['deletion_matrix'], axis=0)

    # Add all_atom_mask and dummy all_atom_positions based on aatype.
    all_atom_mask = residue_constants.STANDARD_ATOM_MASK[sequence_features['aatype']]
    msa_features['all_atom_mask'] = all_atom_mask
    msa_features['all_atom_positions'] = np.zeros(list(all_atom_mask.shape) + [3])

    # Add assembly_num_chains.
    msa_features['assembly_num_chains'] = np.asarray(num_chains)

    # Add entity_mask.
    msa_features['entity_mask'] = (sequence_features['entity_id'] != 0).astype(np.int32)

    # Fix so that num_alignments is correct shape
    msa_features['num_alignments'] = np.array(msa_features['msa'].shape[0],dtype=np.int32)

    # Create template features
    # TEMPLATE_FEATURES = {
    #         "template_aatype": np.float32,
    #         "template_all_atom_positions": np.float32,
    #         "template_domain_names": np.object,
    #         "template_sequence": np.object,
    #         "template_sum_probs": np.float32,
    #         }
    # template_features = {}
    # merged_len = sequence_features['aatype'].shape[0]
    # template_features["num_templates"] = np.asarray(1, dtype=np.int32),                       
    # template_features["template_all_atom_mask"] = np.zeros((1, merged_len, 37), dtype=np.int32)            
    # template_features["template_all_atom_positions"] = np.zeros((1, merged_len, 37, 3), dtype=np.float32)
    # for name in TEMPLATE_FEATURES:
    #     if name not in ["num_templates", "template_all_atom_mask", "template_all_atom_positions"]:
    #         template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])
    #         templates_result = templates.TemplateSearchResult(
    #             features=template_features, errors=[], warnings=[]
    #         )
    merged_len = sequence_features['aatype'].shape[0]
    template_features = {
          'template_aatype': np.zeros(
              (1, merged_len),
              dtype=np.int32),
          'template_all_atom_mask': np.zeros(
              (1, merged_len, residue_constants.atom_type_num), dtype=np.int32),
          'template_all_atom_positions': np.zeros(
              (1, merged_len, residue_constants.atom_type_num, 3), np.float32),
          'template_domain_names': np.array([''.encode()], dtype=np.object),
          'template_sequence': np.array([''.encode()], dtype=np.object),
          'template_sum_probs': np.array([0], dtype=np.float32)
    }


    # TEMPLATE_FEATURES = {
    #         "template_aatype": np.float32,
    #         "template_all_atom_masks": np.float32,
    #         "template_all_atom_positions": np.float32,
    #         "template_domain_names": np.object,
    #         "template_sequence": np.object,
    #         "template_sum_probs": np.float32,
    #     }
    # template_features = {}
    # for name in TEMPLATE_FEATURES:
    #     template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])
    #     templates_result = templates.TemplateSearchResult(
    #         features=template_features, errors=[], warnings=[]
    #     )
    

    # Crop msas and templates before creating masks
    features = _crop_single_chain({**msa_features, **template_features, **sequence_features}, msa_crop_size=2048, pair_msa_sequences=False, max_templates=1)

    # create cluster_bias_mask
    # mask = np.zeros(msa_features['msa'].shape[0])
    # mask[0] = 1
    # features['cluster_bias_mask'] = mask

    mask = np.zeros(features['msa'].shape[0])
    mask[0] = 1
    features['cluster_bias_mask'] = mask
    
    # Initialize Bert mask with masked out off diagonals.
    msa_masks = [np.ones(features['msa'].shape, dtype=np.float32)]
    features['bert_mask'] = block_diag(*msa_masks, pad_value=0)
    
    #sys.exit()
    features['seq_length'] = np.asarray(features['aatype'].shape[0],
                                        dtype=np.int32)
    features['num_alignments'] = np.asarray(features['msa'].shape[0],
                                                dtype=np.int32)
    

    features = process_final(features)
        

    for n, msa in enumerate(msa_objects):
        logging.info('MSA %d size: %d sequences.', n, len(msa.sequences))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'])
    return features#{**msa_features, **merged_sequence_features, **templates_result.features}
  



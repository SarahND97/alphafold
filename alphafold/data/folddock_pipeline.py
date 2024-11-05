import os, sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.pipeline import make_msa_features, make_sequence_features, FeatureDict
from alphafold.data.pipeline_multimer import convert_monomer_features, pad_msa
from alphafold.data.feature_processing import _crop_single_chain, process_final
from alphafold.data.msa_pairing import block_diag, SEQ_FEATURES
import numpy as np

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
    # Pad MSA to avoid zero-sized extra_msa.
    features = pad_msa(features, 512)
        
    # for n, msa in enumerate(msa_objects):
    #     logging.info('MSA %d size: %d sequences.', n, len(msa.sequences))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'])
    return features#{**msa_features, **merged_sequence_features, **templates_result.features}
  



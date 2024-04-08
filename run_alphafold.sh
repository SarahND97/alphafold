documents=/home/sarahnd/Documents
singularity_img=$documents/evoformer_alphafold/singularity-image/alphafold_v2.3.1.sif
disk=/media/sarahnd/f4aa44bb-cc6e-486a-bc4c-7b4f1193d8b4/msas_2018_benchmark
# Choose one of
# heteromers  homomers  negative_heteromers  neg_homodimers
type=heteromers
msas=$disk/$type
pdbid=5oxz
msas_specific=$msas/$pdbid/msas
output=$disk/pkl-files
FASTA_PATH=$documents/D-SCRIPT/data/$type

$singularity_img python3 run_alphafold.py \
    --fasta_paths=$FASTA_PATH/$pdbid.fasta \
    --model_preset=multimer \
    --msa_dir=$msas_specific \
    --output_dir=$output \
    --data_dir=$PARAM \
    --use_gpu_relax=False 


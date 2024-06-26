home=/home/sarahnd
documents=$home/Documents
evoformer=$documents/evoformer_alphafold/
singularity_img=$evoformer/singularity-image/alphafold_v2.3.1.sif
disk=$home/mnt/sarahnd/msas_2018_benchmark
# Choose one of
# heteromers  homomers  negative_heteromers  neg_homodimers
type=heteromers
msas=$disk/$type
#pdbid=5oxz
#msas_specific=$msas/$pdbid/msas
FASTA_PATH=$documents/D-SCRIPT/data/seqres/$type
alpha_dir=$evoformer/alphafold
PARAMS=$alpha_dir/afdb
output=$disk/pkl-files
pdbids_file=$disk/heteromers/ids_heteromers.txt
lines_to_read=30

# Read the first 30 lines of the file and perform an action on each line
while IFS= read -r pdbid && [ $lines_to_read -gt 0 ]; do
    singularity exec --nv --bind $home:$home $singularity_img \
    python3 $alpha_dir/run_alphafold.py \
        --fasta_paths=$FASTA_PATH/$pdbid.fasta \
        --model_preset=multimer \
        --msa_dir=$msas/$pdbid/msas \
        --output_dir=$output \
        --data_dir=$PARAMS \
        --use_gpu_relax=False \
        --num_multimer_predictions_per_model=1 \
        --separate_output_dir=False \
        --model_to_run="random" \
done < "$pdbids_file"



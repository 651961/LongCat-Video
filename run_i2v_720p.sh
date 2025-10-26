# 480p_distill
torchrun --nproc_per_node=8 /codes/LongCat-Video/480p_distill.py \
    --csv_path /codes/data.csv \
    --checkpoint_dir /models/LongCat-Video \
    --stage1_output_dir /codes/output_480p_distill \
    --context_parallel_size 8 \
    --seed 42 \
    --enable_compile

# 720p_refinement
torchrun --nproc_per_node=8 /codes/LongCat-Video/720p_refinement.py \
    --csv_path /codes/data.csv \
    --checkpoint_dir /models/LongCat-Video \
    --stage1_output_dir /codes/output_480p_distill \
    --final_output_dir /codes/output_720p_refinement \
    --context_parallel_size 8 \
    --seed 42 \
    --enable_compile
python eval/eval_ppl.py --hf_path meta-llama/Llama-2-7b-hf \
 --output_dir ./results/ppls \
 --replace_method complex_phase_v2 \
 --cuda 1
# --skip_lm_head (False by default for Fairy2i-W2, i.e., lm_head will be replaced)
 
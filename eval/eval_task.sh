python eval/eval_task.py \
  --hf_path meta-llama/Llama-2-7b-hf \
  --batch_size 8 \
  --tasks arc_easy,arc_challenge,hellaswag,boolq,openbookqa,piqa,winogrande \
  --output_path results/tasks/example_task_results.json \
  --num_fewshot 0 \
  --replace_method complex_phase_v2 \
  --cuda 0 \
  --ctx_size 2048\
  # --skip_lm_head (False by default for Fairy2i-W2, i.e., lm_head will be replaced)
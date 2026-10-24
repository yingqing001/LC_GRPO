# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_batch.log 2>&1 & #No.2

# single-node:
# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.6_steps8.log 2>&1 & #No.4

# multi-node (run on EACH node with its own NODE_RANK):
#   gpu-h200-23 (master):  NODE_RANK=0 bash test.sh
#   gpu-h200-06 (worker):  NODE_RANK=1 bash test.sh
# Defaults now match this cluster (NCCL_SOCKET_IFNAME=bond0, MASTER_ADDR=10.2.201.23),
# so no need to pass $IF / MASTER_ADDR. Override only if a node differs:
#   NODE_RANK=1 NCCL_SOCKET_IFNAME=<iface> MASTER_ADDR=<node0-ip> bash test.sh


# nohup bash hunyuan_grpo_correct_multinode.sh \
#     > output_log/base_const_noise0.6_steps8_rank${NODE_RANK:-0}.log 2>&1 & #No.3,4




# nohup bash hunyuan_grpo_improved.sh > output_log/cps_noise0.7_steps16.log 2>&1 & #No.3

# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.6_steps12.log 2>&1 & #No.5

# nohup bash hunyuan_grpo_correct.sh > output_log/ddim_noise0.7_steps8.log 2>&1 & #No.1


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.5_steps8.log 2>&1 & #No.2


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.7_steps12.log 2>&1 & #No.5


# nohup bash hunyuan_grpo_correct.sh > output_log/ddim_noise0.8_steps8.log 2>&1 & #No.4



# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.6_steps8_rep.log 2>&1 & #No.4


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.7_steps8_rep.log 2>&1 & #No.5



# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_rep.log 2>&1 & #No.3



# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.6_steps8_prompt32.log 2>&1 & #No.2


# nohup bash hunyuan_grpo_improved.sh > output_log/cps_noise0.7_steps16_prompt32.log 2>&1 & #No.1


# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_prompt32.log 2>&1 & #No.4


# nohup bash hunyuan_grpo_improved.sh > output_log/cps_noise0.8_steps16_prompt32.log 2>&1 & #No.1


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.7_steps8_prompt32.log 2>&1 & #No.5


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.8_steps8_prompt32.log 2>&1 & #No.3


# nohup bash hunyuan_grpo_improved.sh > output_log/cps_noise0.8_steps16_prompt16.log 2>&1 & #No.4


# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_bs_1.log 2>&1 & #No.2
# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_bs_8.log 2>&1 & #No.2


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.8_steps8_prompt16_VMT.log 2>&1 & #No.3
# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.8_steps8_prompt16_M.log 2>&1 & #No.4
# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.7_steps8_prompt16_M.log 2>&1 & #No.5

# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_noise0.8_steps8_prompt16_V0.4_M0.4_T0.2.log 2>&1 & #No.2


# nohup bash hunyuan_grpo_correct.sh > output_log/base_const_test.log 2>&1 & #No.5
# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_dance_grpo_test.log 2>&1 & #No.5
# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_cps_test.log 2>&1 & #No.5

# nohup bash hunyuan_grpo_orig_batch.sh > output_log/baseline_orig_batch_noise0.25_sample16_prompt16_VMT.log 2>&1 & #No.5
# nohup bash hunyuan_grpo_orig.sh > output_log/baseline_orig_noise0.25_sample16_prompt16_VMT.log 2>&1 & #No.5

# nohup bash hunyuan_grpo_improved.sh > output_log/baseline_cps_noise0.8_steps16_prompt16_VMT.log 2>&1 & #No.4

# nohup bash hunyuan_grpo_orig.sh > output_res_log/sde_noise0.25_sample16_prompt16_VMT.log 2>&1 & #No.5
# nohup bash hunyuan_grpo_orig_cps.sh > output_res_log/cps_noise0.8_sample16_prompt16_VMT.log 2>&1 & #No.4
# nohup bash hunyuan_grpo_orig_correct.sh > output_res_log/correct_noise0.8_sample8_prompt16_VMT.log 2>&1 & #No.3

# nohup bash hunyuan_grpo_orig.sh > output_res_log/sde_noise0.25_sample16_prompt16_V.log 2>&1 & #No.2

# nohup bash generate_vbench.sh > output_res_log/vbench_base_model.log 2>&1 & #No.2

# nohup bash hunyuan_grpo_orig_correct.sh > output_res_log/correct_noise0.7_sample8_prompt16_VMT.log 2>&1 & #No.4


# nohup bash eval_demo32.sh > output_res_log/eval_demo_step50.log 2>&1 & #No.2

# nohup bash eval_vbench.sh > output_res_log/eval_vbench_part1.log 2>&1 & #No.5
# nohup bash eval_vbench_2.sh > output_res_log/eval_vbench_part2.log 2>&1 & #No.3


# nohup bash eval_VGenEval.sh > output_res_log/eval_vgeneval.log 2>&1 & #No.5
# nohup bash eval_VGenEval.sh > output_res_log/eval_vgeneval_full.log 2>&1 & #No.2


# nohup bash eval_vbench_3.sh > output_res_log/eval_vbench_part3.log 2>&1 & #No.4
# nohup bash eval_vbench_4.sh > output_res_log/eval_vbench_part6.log 2>&1 & #No.3

# nohup bash eval_vbench.sh > output_res_log/eval_vbench_part5.log 2>&1 & #No.5


# nohup bash eval_vbench_3.sh > output_res_log/eval_vbench_part7.log 2>&1 & #No.2

# nohup bash eval_demo32.sh > output_res_log/eval_demo_step50.log 2>&1 & #No.2

# Run the rollouts SEQUENTIALLY (each uses all 8 GPUs, so they can't share the node).
# The loop body has no '&', so each torchrun blocks until it finishes; the next one
# starts only after the previous returns. Launch in the background with:
#   nohup bash test.sh > output_log/rollout_both.log 2>&1 &

# mkdir -p output_log
# nohup bash -c '
# for job in rollout_hunyuan_orig_correct rollout_hunyuan_orig; do
#     echo "===== starting ${job} at $(date) ====="
#     bash ${job}.sh > output_log/${job}.log 2>&1
#     echo "===== finished ${job} (exit $?) at $(date) ====="
# done
# ' > output_log/rollout_all.log 2>&1 &


# bash eval_demo32.sh > output_res_log/eval_demo_step50.log 2>&1 & #No.5
# bash eval_demo32.sh > output_res_log/eval_demo_step50_1.log 2>&1 & #No.2


# nohup bash eval_vbench_2.sh > output_res_log/eval_vbench.log 2>&1 & #No.3
# nohup bash eval_vbench_3.sh > output_res_log/eval_vbench_part1.log 2>&1 & #No.2
# nohup bash eval_vbench_4.sh > output_res_log/eval_vbench_part2.log 2>&1 & #No.4


# nohup bash hunyuan_grpo_orig_correct.sh > output_res_log/correct_noise0.6_sample8_prompt16_V.log 2>&1 & #No.4
# nohup bash hunyuan_grpo_orig_correct.sh > output_res_log/correct_noise0.7_sample8_prompt16_V.log 2>&1 & #No.5

# nohup bash hunyuan_grpo_correct.sh > output_res_log/correct_noise0.6_sample8_prompt16_V_v1.log 2>&1 & #No.4
# nohup bash hunyuan_grpo_correct.sh > output_res_log/correct_noise0.7_sample8_prompt16_V_v1.log 2>&1 & #No.5


# nohup bash eval_100.sh > output_res_log/eval_100.log 2>&1 & #No.4

# bash eval_demo32.sh > output_res_log/eval_demo_step50.log 2>&1 & #No.5
# bash eval_demo32.sh > output_res_log/eval_demo_step50_1.log 2>&1 & #No.4

# nohup bash eval_vbench_2.sh > output_res_log/eval_vbench.log 2>&1 & #No.4
# nohup bash eval_vbench_3.sh > output_res_log/eval_vbench_part3.log 2>&1 & #No.5


# nohup bash run_vbench.sh > output_res_log/eval_vbench.log 2>&1 & #No.5
# nohup bash run_vbench_2.sh > output_res_log/eval_vbench_2.log 2>&1 & #No.4

# nohup bash eval_demo32_visual.sh > output_res_log/eval_demo_step50.log 2>&1 & #No.5
nohup bash eval_demo32_visual.sh > output_res_log/eval_demo_step50_1.log 2>&1 & #No.4
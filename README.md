## News
- [2025/04/06] We released a novel RL algorithm named Trust Region Preference Approximation (TRPA) for LLM reasoning tasks on 6 Apr 2025, welcome your suggestion and discussion! Paper link: [Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning](https://arxiv.org/abs/2504.04524).

# Reproduce-DeepSeek-R1-Survey
This repository collects various works that reproduce DeepSeek R1, as well as works related to DeepSeek R1 and the DeepSeek series.

## Work related to DeepSeek's Tech, i.e. papers for LLM reasoning tasks mainly relying on RL
1. Logic RL: R1 reproduction of logic problems https://arxiv.org/abs/2502.14768
2. Start RL with short context length and gradually increase context length. https://github.com/agentica-project/deepscaler 
3. The role of CoT in Reasoning https://github.com/eddycmu/demystify-long-cot  https://arxiv.org/pdf/2502.03373
4. Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning https://github.com/XueruiSu/Trust-Region-Preference-Approximation
5. Hierarchical LLM Reasoning via Scaling Thought Templates Revolutionary inference-scaling paradigm with a hierarchical RL algorithm: empowering a 32B model with 500 thought templates to outperform o1-preview and DeepSeek-V3 in reasoning tasks. https://github.com/Gen-Verse/ReasonFlux 
6. By imitating the correct samples, learning the preference of the wrong samples, and focusing on the key steps, there is no need to rely on super-large-scale models (such as DeepSeek-R1) for distillation, and only training is done through reinforcement learning. 
 https://github.com/InternLM/OREAL https://arxiv.org/abs/2502.06781 
7. Build a structured thinking template library. This idea should be more suitable for Science problems. Reasoning for Science problems can be done in this way. https://github.com/Gen-Verse/ReasonFlux https://arxiv.org/abs/2502.06772 
8. https://github.com/Jiayi-Pan/TinyZero
9. Multi-Turn System: Teach large language models (LLMs) to criticize and improve their outputs, and train critics through reinforcement learning.  https://arxiv.org/abs/2502.0349 
10. Experience the Ahah moment yourself for < $30. Through RL, the 3B base LM develops self-verification and search abilities https://github.com/Jiayi-Pan/TinyZero
11. Only ORM does Rewardmodel+RL for training  https://github.com/Mohammadjafari80/GSM8K-RLVR
12. Agent Training Gym-Sokoban https://github.com/ZihanWang314/RAGEN
13. R1 Computer Use (LLM-Computer Interaction) train an agent to interact with a computer environment (e.g., file system, web browser, command line) while utilizing a neural reward model to validate the correctness of the agent’s actions and reason about intermediate steps.    https://github.com/agentsea/r1-computer-use 
14. Aha Moment reproduction tutorial.  https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb  
15. Simple Reinforcement Learning for Reasoning  https://github.com/hkust-nlp/simpleRL-reason
16. Open-Reasoner-Zero https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
17. MLA Transfer https://arxiv.org/abs/2502.14837 https://github.com/JT-Ushio/MHA2MLA
18. Online-DPO-R1 https://github.com/RLHFlow/Online-DPO-R1
19. Short-RL https://github.com/lblankl/Short-RL
20. Seed-Thinking-v1.5 https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/
21. Skywork-OR1 https://github.com/SkyworkAI/Skywork-OR1

### How to get more beneficial responses for RL training during inference?
1. TTRL https://arxiv.org/abs/2504.16084
2. DAPO: DAPO: An Open-Source LLM Reinforcement Learning System at Scale https://arxiv.org/abs/2503.14476
3. Dr.GRPO https://arxiv.org/abs/2503.20783
4. CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models https://arxiv.org/abs/2503.22342

## Multi-Modal Reasoning Large Language Model
1. BioMedGPT-R1  https://www.163.com/dy/article/JOU1ULKJ0511B8LM.html https://finance.sina.com.cn/tech/digi/2025-02-21/doc-inemfmwk1568534.shtml
2. Kimi-k1.5 https://github.com/MoonshotAI/Kimi-k1.5
3. UNO https://arxiv.org/abs/2504.02160  https://bytedance.github.io/UNO/
4. KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking https://github.com/juyeonnn/KGMEL
5. 

## Other work that is different from DeepSeek's technical route but is equally "Aha":
1. With only 817 training samples, it achieves outstanding performance on the AIME and MATH benchmarks. Belief: After the model has accumulated rich knowledge during the pre-training phase, it may only need a small number of well-structured samples to unlock advanced reasoning capabilities. https://github.com/GAIR-NLP/LIMO 
2. Rule Based Rewards for Language Model Safety https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf
3. MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations.  https://arxiv.org/abs/2502.06453
4. Free Process Rewards without Process Labels. https://arxiv.org/abs/2412.01981
5. Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback. https://arxiv.org/abs/2501.12895
6. Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling https://arxiv.org/abs/2502.06703
7. MedS3: Towards Medical Small Language Models with Self-Evolved Slow Thinking. MedS3 consists of a policy model and a process reward model (PRM), which is iteratively optimized by learning on 16 different datasets, including medical diagnosis, biomedicine, and knowledge question answering, using only 7465 seed data, combined with fine-grained Monte Carlo tree search and process supervision signals for rule verification. https://arxiv.org/pdf/2501.12051
8. s1: Simple test-time scaling s1K contains 1000 carefully selected math problems and reasoning traces distilled from Gemini Flash. When selecting problems, researchers focus on difficulty, diversity, and quality. By fine-tuning Qwen2.5-32B-Instruct on the s1K dataset, the researchers successfully surpassed OpenAI's o1-preview in the competition math benchmark, with a maximum improvement of 27%. https://arxiv.org/abs/2501.19393
9. Learning Adaptive **Parallel** Reasoning with Language Models https://github.com/Parallel-Reasoning/APR  **(amazing idea for different reasoning tech)**

## cost-effectively Reinforcement Learning
1. Tina: Tiny Reasoning Models via LoRA https://github.com/shangshang-wang/Tina
   
## Reasoning Dataset
1.  open-r1/OpenR1-Math-220k https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
2. Congliu/Chinese-DeepSeek-R1-Distill-data-110k https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k
3. Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT
4. TechxGenus/deepseek_r1_code_1k https://huggingface.co/datasets/TechxGenus/deepseek_r1_code_1k
5. umarigan/deepseek-r1-reasoning-prompts https://huggingface.co/datasets/umarigan/deepseek-r1-reasoning-prompts
6. mlabonne/dolphin-r1-deepseek mlabonne/dolphin-r1-deepseek
7. DKYoon/dolphin-r1-deepseek-filtered-short https://huggingface.co/datasets/DKYoon/dolphin-r1-deepseek-filtered-short
8. open-llm-leaderboard/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B-details https://huggingface.co/datasets/open-llm-leaderboard/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B-details
9. agentica-org/DeepScaleR-Preview-Dataset: https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
10. 2K logic dataset: https://huggingface.co/datasets/K-and-K/knights-and-knaves   which used in Logic RL (https://github.com/Unakar/Logic-RL)
11. Open-Reasoner-Zero/orz_math_57k_collection https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection
12. Beyond AIME (Seed-Thinking-v1.5) https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/
13. Skywork/Skywork-OR1-RL-Data https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data

## PipeLine work which can be used to reproduce R1 more easily:
1. HybridFlow: A Flexible and Efficient RLHF Framework. https://arxiv.org/abs/2409.19256 https://github.com/volcengine/verl
2. ReaLHF: ReaL: Efficient RLHF Training for LLMs with Parameter Reallocation. https://github.com/openpsi-project/ReaLHF
3. OpenRLHF: a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers. https://github.com/OpenRLHF/OpenRLHF
4. DeepSpeed Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat/ds-chat-release-8-31 / https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md
5. NVIDIA NeMo-Aligner: a scalable toolkit for efficient model alignment. https://github.com/NVIDIA/NeMo-Aligner
6. MoBA: MIXTURE OF BLOCK ATTENTION FOR LONG-CONTEXT LLMS https://github.com/MoonshotAI/MoBA
7. Open R1 https://github.com/huggingface/open-r1

## DeepSeek Series
1. DreamCraft3D https://arxiv.org/abs/2310.16818
2. Fire-Flyer AI-HPC https://arxiv.org/abs/2408.14158
3. Expert-Specialized Fine-Tuning https://github.com/deepseek-ai/ESFT
4. Janus-Pro https://github.com/deepseek-ai/Janus https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf
5. DeepSeek LLM https://arxiv.org/pdf/2401.02954
6. DeepSeekMoE https://arxiv.org/abs/2401.06066
7. DeepSeekMath https://arxiv.org/abs/2402.03300
8. DeepSeek-V2 https://arxiv.org/abs/2405.04434
9. DeepSeek-V3 https://arxiv.org/html/2412.19437v1
10. DeepSeek-R1 https://arxiv.org/abs/2501.12948
11. DeepSeek-Coder-V2 https://github.com/deepseek-ai/DeepSeek-Coder-V2  https://arxiv.org/pdf/2406.11931
12. CodeI/O https://arxiv.org/pdf/2502.07316
13. Native Sparse Attention https://arxiv.org/abs/2502.11089
14. DeepSeek-VL2 https://arxiv.org/abs/2412.10302
15. DeepSeek-Prover-V1.5 https://arxiv.org/abs/2408.08152
16. DeepSeek-Prover https://arxiv.org/abs/2405.14333
17. DeepSeek-VL https://arxiv.org/abs/2403.05525
18. DeepSeek-Coder https://arxiv.org/abs/2401.14196
19. DeepSeek: Content Based Image Search & Retrieval https://arxiv.org/abs/2401.14196
20. FlashMLA https://github.com/deepseek-ai/FlashMLA
21. DeepEP https://github.com/deepseek-ai/DeepEP
22. DeepGEMM https://github.com/deepseek-ai/DeepGEMM

    
## Discussion about Reward System
1. Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning https://arxiv.org/abs/2502.06781










# Reproduce-DeepSeek-R1-Survey
This repository collects various works that reproduce DeepSeek R1, as well as works related to DeepSeek R1 and the DeepSeek series.

## Work which reproduce R1
1. R1 reproduction of logic problems https://github.com/Unakar/Logic-RL 
2. Start RL with short context length and gradually increase context length. https://github.com/agentica-project/deepscaler 
3. The role of CoT in Reasoning https://github.com/eddycmu/demystify-long-cot  https://arxiv.org/pdf/2502.03373
4. s1: Simple test-time scaling s1K contains 1000 carefully selected math problems and reasoning traces distilled from Gemini Flash. When selecting problems, researchers focus on difficulty, diversity, and quality. By fine-tuning Qwen2.5-32B-Instruct on the s1K dataset, the researchers successfully surpassed OpenAI's o1-preview in the competition math benchmark, with a maximum improvement of 27%. https://arxiv.org/abs/2501.19393 
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
16. open-r1/OpenR1-Math-220k https://huggingface.co/datasets/open-r1/OpenR1-Math-220k



## Other work that is different from DeepSeek's technical route but is equally "Aha":
1. With only 817 training samples, it achieves outstanding performance on the AIME and MATH benchmarks. Belief: After the model has accumulated rich knowledge during the pre-training phase, it may only need a small number of well-structured samples to unlock advanced reasoning capabilities. https://github.com/GAIR-NLP/LIMO 
2. Rule Based Rewards for Language Model Safety https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf
3. MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations.  https://arxiv.org/abs/2502.06453
4. Free Process Rewards without Process Labels. https://arxiv.org/abs/2412.01981
5. Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback. https://arxiv.org/abs/2501.12895
6. Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling https://arxiv.org/abs/2502.06703
   


## PipeLine work which can be used to more easily reproduce R1:
1. HybridFlow: A Flexible and Efficient RLHF Framework. https://arxiv.org/abs/2409.19256 https://github.com/volcengine/verl
2. ReaLHF: ReaL: Efficient RLHF Training for LLMs with Parameter Reallocation. https://github.com/openpsi-project/ReaLHF
3. OpenRLHF: a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers. https://github.com/OpenRLHF/OpenRLHF
4. DeepSpeed Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat/ds-chat-release-8-31 / https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md
5. NVIDIA NeMo-Aligner: a scalable toolkit for efficient model alignment. https://github.com/NVIDIA/NeMo-Aligner
6. MoBA: MIXTURE OF BLOCK ATTENTION FOR LONG-CONTEXT LLMS https://github.com/MoonshotAI/MoBA
7. Kimi-k1.5 https://github.com/MoonshotAI/Kimi-k1.5

## DeepSeek Series
1. DeepSeek LLM https://arxiv.org/pdf/2401.02954
2. DeepSeekMath https://arxiv.org/abs/2402.03300
3. DeepSeek-V2 https://arxiv.org/abs/2405.04434
4. DeepSeek-V3 https://arxiv.org/html/2412.19437v1
5. DeepSeek-R1 https://arxiv.org/abs/2501.12948
6. DeepSeek-Coder-V2 https://github.com/deepseek-ai/DeepSeek-Coder-V2  https://arxiv.org/pdf/2406.11931
7. CodeI/O https://arxiv.org/pdf/2502.07316
8. Native Sparse Attention https://arxiv.org/abs/2502.11089
9. DeepSeek-VL2 https://arxiv.org/abs/2412.10302
10. DeepSeek-Prover-V1.5 https://arxiv.org/abs/2408.08152
11. DeepSeek-Prover https://arxiv.org/abs/2405.14333
12. DeepSeek-VL https://arxiv.org/abs/2403.05525
13. DeepSeek-Coder https://arxiv.org/abs/2401.14196
14. DeepSeek: Content Based Image Search & Retrieval https://arxiv.org/abs/2401.14196
    











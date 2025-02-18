# Reproduce-DeepSeek-R1-Survey
This repository collects various works that reproduce DeepSeek R1, as well as works related to DeepSeek R1 and the DeepSeek series.

## Work which reproduce R1
1. https://github.com/Unakar/Logic-RL R1 reproduction of logic problems
Qwen2-7B-Instruct, 2K Logic Puzzle Dataset
2. https://github.com/agentica-project/deepscaler Start RL with short context length and gradually increase context length.
3. https://github.com/eddycmu/demystify-long-cot The role of CoT in Reasoning https://arxiv.org/pdf/2502.03373
4. https://arxiv.org/abs/2501.19393 s1: Simple test-time scaling s1K contains 1000 carefully selected math problems and reasoning traces distilled from Gemini Flash. When selecting problems, researchers focus on difficulty, diversity, and quality. By fine-tuning Qwen2.5-32B-Instruct on the s1K dataset, the researchers successfully surpassed OpenAI's o1-preview in the competition math benchmark, with a maximum improvement of 27%.
5. https://github.com/Gen-Verse/ReasonFlux Hierarchical LLM Reasoning via Scaling Thought Templates Revolutionary inference-scaling paradigm with a hierarchical RL algorithm: empowering a 32B model with 500 thought templates to outperform o1-preview and DeepSeek-V3 in reasoning tasks.
6. https://github.com/InternLM/OREAL https://arxiv.org/abs/2502.06781 By imitating the correct samples, learning the preference of the wrong samples, and focusing on the key steps, there is no need to rely on super-large-scale models (such as DeepSeek-R1) for distillation, and only training is done through reinforcement learning.
7. https://github.com/Gen-Verse/ReasonFlux https://arxiv.org/abs/2502.06772 Core: Build a structured thinking template library. This idea should be more suitable for Science problems. Reasoning for Science problems can be done in this way.
8. https://github.com/Jiayi-Pan/TinyZero
9. Multi-Turn System: Teach large language models (LLMs) to criticize and improve their outputs, and train critics through reinforcement learning. https://arxiv.org/abs/2502.0349 (bytes)
10. https://github.com/Jiayi-Pan/TinyZero Experience the Ahah moment yourself for < $30. Through RL, the 3B base LM develops self-verification and search abilities
11. https://github.com/Mohammadjafari80/GSM8K-RLVR Only ORM does Rewardmodel+RL for training
12. Agent Training Gym-Sokoban: https://github.com/ZihanWang314/RAGEN
13. R1 Computer Use (LLM-Computer Interaction) train an agent to interact with a computer environment (e.g., file system, web browser, command line) while utilizing a neural reward model to validate the correctness of the agent’s actions and reason about intermediate steps. https://github.com/agentsea/r1-computer-use
14. Aha Moment reproduction tutorial: https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb
15. https://github.com/hkust-nlp/simpleRL-reason Simple Reinforcement Learning for Reasoning



## Other work that is different from DeepSeek's technical route but is equally "Aha":
1. https://github.com/GAIR-NLP/LIMO With only 817 training samples, it achieves outstanding performance on the AIME and MATH benchmarks. Belief: After the model has accumulated rich knowledge during the pre-training phase, it may only need a small number of well-structured samples to unlock advanced reasoning capabilities.
2. Rule Based Rewards for Language Model Safety https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf
3. MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations.  https://arxiv.org/abs/2502.06453
4. Free Process Rewards without Process Labels. https://arxiv.org/abs/2412.01981
5. Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback. https://arxiv.org/abs/2501.12895 


## PipeLine work which can be used to more easily reproduce R1:
1. HybridFlow: A Flexible and Efficient RLHF Framework. https://arxiv.org/abs/2409.19256 https://github.com/volcengine/verl
2. ReaLHF: ReaL: Efficient RLHF Training for LLMs with Parameter Reallocation. https://github.com/openpsi-project/ReaLHF
3. OpenRLHF: a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers. https://github.com/OpenRLHF/OpenRLHF
4. DeepSpeed Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat/ds-chat-release-8-31 / https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md
5. NVIDIA NeMo-Aligner: a scalable toolkit for efficient model alignment. https://github.com/NVIDIA/NeMo-Aligner


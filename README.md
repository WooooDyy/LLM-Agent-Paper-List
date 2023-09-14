# The Rise and Potential of Large Language Model Based Agents: A Survey
🔥 **Must-read papers for LLM-based agents.**

## 🔔 News
- [2023/09/14] We will release our survey paper: Large Language Model Based Agents: The Past, Present, and Potential . 
  - You can also download it here:
- [2023/09/14] We create this repository to introduce a survey of LLM-based agents and list some must-read papers. More papers will be added soon!

<div align=center><img src="./assets/figure1.jpg" width="80%" /></div>


## 🌟 Introduction

For a long time, humanity has pursued artificial intelligence (AI) equivalent to or surpassing human level, with AI agents considered as a promising vehicle of this pursuit. AI agents are artificial entities that sense their environment, make decisions, and take actions. 

Due to the versatile and remarkable capabilities they demonstrate, large language models (LLMs) are regarded as potential sparks for Artificial General Intelligence (AGI), offering hope for building general AI agents. Many research efforts have leveraged LLMs as the foundation to build AI agents and have achieved significant progress.

In this repository, we provide a systematic and comprehensive survey on LLM-based agents, and list some must-read papers. 

Specifically, we start by the general conceptual framework for LLM-based agents: comprising three main components: brain, perception, and action, and the framework can be tailored to suit different applications. 
Subsequently, we explore the extensive applications of LLM-based agents in three aspects: single-agent scenarios, multi-agent scenarios, and human-agent cooperation. 
Following this, we delve into agent societies, exploring the behavior and personality of LLM-based agents, the social phenomena that emerge when they form societies, and the insights they offer for human society.
Finally, we discuss a range of key topics and open problems within the field.

**We greatly appreciate any contributions via PRs, issues, emails, or other methods.**



- [The Rise and Potential of Large Language Model Based Agents: A Survey](#the-rise-and-potential-of-large-language-model-based-agents-a-survey)
  - [🔔 News](#-news)
  - [🌟 Introduction](#-introduction)
  - [1. The Birth of An Agent: Construction of LLM-based Agents](#1-the-birth-of-an-agent-construction-of-llm-based-agents)
    - [1.1 Brain: Primarily Composed of An LLM](#11-brain-primarily-composed-of-an-llm)
    - [1.2 Perception: Multimodal Inputs for LLM-based Agents](#12-perception-multimodal-inputs-for-llm-based-agents)
    - [1.3 Action: Expand Action Space of LLM-based Agents](#13-action-expand-action-space-of-llm-based-agents)
      - [Tool Using](#tool-using)
      - [Embodied Action](#embodied-action)
  - [2. Agents in Practice: Applications of LLM-based Agents](#2-agents-in-practice-applications-of-llm-based-agents)
    - [2.1 General Ability of Single Agent](#21-general-ability-of-single-agent)
    - [2.2 Coordinating Potential of Multiple Agents](#22-coordinating-potential-of-multiple-agents)
    - [2.3 Interactive Engagement between Human and Agent](#23-interactive-engagement-between-human-and-agent)
      - [2.3.1 Instructor-Executor Paradigm](#231-instructor-executor-paradigm)
        - [Education](#education)
        - [Health](#health)
        - [Other Application](#other-application)
      - [2.3.2 Equal Partnership Paradigm](#232-equal-partnership-paradigm)
        - [Empathetic Communicator](#empathetic-communicator)
        - [Human-Level Participant](#human-level-participant)
  - [3. Agent Society: From Individuality to Sociality](#3-agent-society-from-individuality-to-sociality)
    - [3.1 Behavior and Personality of LLM-based Agent](#31-behavior-and-personality-of-llm-based-agent)
    - [3.2 Environment for Agent Society](#32-environment-for-agent-society)
    - [3.3 Society Simulation with LLM-based Agents](#33-society-simulation-with-llm-based-agents)
  - [Project Maintainers \& Contributors](#project-maintainers--contributors)
  - [Contact](#contact)





## 1. The Birth of An Agent: Construction of LLM-based Agents
<div align=center><img src="./assets/figure2.jpg" width="60%" /></div>

### 1.1 Brain: Primarily Composed of An LLM

#### Knowledge

##### Pretrain model

- [2023/04] **Learning Distributed Representations of Sentences from Unlabelled Data.** *Felix Hill(University of Cambridge) et al. arXiv.*[[paper](https://arxiv.org/abs/1602.03483)]
- [2020/02] **How Much Knowledge Can You Pack Into the Parameters of a Language Model?** *Adam Roberts(Google) et al. arXiv.*[[paper](https://arxiv.org/abs/2002.08910)]
- [2020/01] **Scaling Laws for Neural Language Models.** *Jared Kaplan(Johns Hopkins University) et al. arXiv.*[[paper](https://arxiv.org/abs/2001.08361)]
- [2017/12] **Commonsense Knowledge in Machine Intelligence.** *Niket Tandon(Allen Institute for Artificial Intelligence) et al. SIGMOD.*[[paper](https://sigmodrecord.org/publications/sigmodRecord/1712/pdfs/09_reports_Tandon.pdf)]
- [2011/03] **Natural Language Processing (almost) from Scratch.** *Ronan Collobert(Princeton) et al. arXiv.*[[paper](https://arxiv.org/abs/1103.0398)]]

##### Linguistic knowledge

- [2023/02] **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity.** *Yejin Bang et al. arXiv.* [[paper](https://arxiv.org/abs/2302.04023)]
- [2021/06] **Probing Pre-trained Language Models for Semantic Attributes and their Values.** *Meriem Beloucif et al. EMNLP.* [[paper](https://aclanthology.org/2021.findings-emnlp.218/)] 
- [2020/10] **Probing Pretrained Language Models for Lexical Semantics.** *Ivan Vulić et al. arXiv.*[[paper](https://arxiv.org/abs/2010.05731)]
- [2019/04] **A Structural Probe for Finding Syntax in Word Representations.** *John Hewitt et al. ACL.*[[paper](https://aclanthology.org/N19-1419/)]
- [2016/04] **Improved Automatic Keyword Extraction Given More Semantic Knowledge.** *H Leung. Systems for Advanced Applications.*[[paper](https://link.springer.com/chapter/10.1007/978-3-319-32055-7_10)]

##### Commensense knowledge

- [2022/10] **Language Models of Code are Few-Shot Commonsense Learners.** *Aman Madaan et al.arXiv.*[[paper](https://arxiv.org/abs/2210.07128)]
- [2021/04] **Relational World Knowledge Representation in Contextual Language Models: A Review.** *Tara Safavi et al. arXiv.*[[paper](https://arxiv.org/abs/2104.05837)]
- [2019/11] **How Can We Know What Language Models Know?** *Zhengbao Jiang et al.arXiv.*[[paper](https://arxiv.org/abs/1911.12543)]

##### Actionable knowledge

- [2023/07] **Large language models in medicine.** *Arun James Thirunavukarasu et al. nature.*[[paper](https://www.nature.com/articles/s41591-023-02448-8)]
- [2023/06] **DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation.** *Yuhang Lai et al. ICML.*[[paper](https://proceedings.mlr.press/v202/lai23b.html)]
- [2022/10] **Language Models of Code are Few-Shot Commonsense Learners.** *Aman Madaan et al. arXiv.*[[paper](https://arxiv.org/abs/2210.07128)]
- [2022/02] **A Systematic Evaluation of Large Language Models of Code.** *Frank F. Xu et al.arXiv.*[[paper](https://arxiv.org/abs/2202.13169)]
- [2021/10] **Training Verifiers to Solve Math Word Problems.** *Karl Cobbe et al. arXiv.* [[paper](https://arxiv.org/abs/2110.14168)]

##### Potential issues of knowledge

- [2023/05] **Editing Large Language Models: Problems, Methods, and Opportunities.** *Yunzhi Yao et al. arXiv.* [[paper](https://arxiv.org/abs/2305.13172)]
- [2023/05] **Self-Checker: Plug-and-Play Modules for Fact-Checking with Large Language Models.** *Miaoran Li et al. arXiv.*[[paper](https://arxiv.org/abs/2305.14623)]
- [2023/05] **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.** *Zhibin Gou et al. arXiv.*[[paper](https://arxiv.org/abs/2305.11738)]
- [2023/04] **Tool Learning with Foundation Models.** *Yujia Qin et al. arXiv.*[[paper](https://arxiv.org/abs/2304.08354)]
- [2023/03] **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.** *Potsawee Manakul et al. arXiv.* [[paper](https://arxiv.org/abs/2303.08896)]
- [2022/06] **Memory-Based Model Editing at Scale.** *Eric Mitchell et al. arXiv.*[[paper](https://arxiv.org/abs/2206.06520)]
- [2022/04] **A Review on Language Models as Knowledge Bases.** *Badr AlKhamissi et al.arXiv.*[[paper](https://arxiv.org/abs/2204.06031)]
- [2021/04] **Editing Factual Knowledge in Language Models.** *Nicola De Cao et al.arXiv.*[[paper](https://arxiv.org/abs/2104.08164)]
- [2017/08] **Measuring Catastrophic Forgetting in Neural Networks.** *Ronald Kemker et al.arXiv.*[[paper](https://arxiv.org/abs/1708.02072)]
### 1.2 Perception: Multimodal Inputs for LLM-based Agents

#### Visual 

- [2023/05] **Language Is Not All You Need: Aligning Perception with Language Models.** *Shaohan Huang et al. arXiv.*[[paper](https://arxiv.org/abs/2302.14045)]]
- [2023/05] **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning.** *Wenliang Dai et al. arXiv.*[[paper](https://arxiv.org/abs/2305.06500)]
- [2023/05] **MultiModal-GPT: A Vision and Language Model for Dialogue with Humans.** *Tao Gong et al. arXiv.*[[paper](https://arxiv.org/abs/2305.04790)]
- [2023/05] **PandaGPT: One Model To Instruction-Follow Them All.** *Yixuan Su et al. arXiv.*[[paper](https://arxiv.org/abs/2305.16355)]
- [2023/04] **Visual Instruction Tuning.** *Haotian Liu et al. arXiv.*[[paper](https://arxiv.org/abs/2304.08485)]
- [2023/04] **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models.** *Deyao Zhu. arXiv.*[[paper](https://arxiv.org/abs/2304.10592)]
- [2023/01] **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** *Junnan Li et al. arXiv.* [[paper](https://arxiv.org/abs/2301.12597)]
- [2022/04] **Flamingo: a Visual Language Model for Few-Shot Learning.** *Jean-Baptiste Alayrac et al. arXiv.*[[paper](https://arxiv.org/abs/2204.14198)]
- [2021/10] **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer.** *Sachin Mehta et al.arXiv.*[[paper](https://arxiv.org/abs/2110.02178)]
- [2021/05] **MLP-Mixer: An all-MLP Architecture for Vision.** *Ilya Tolstikhin et al.arXiv.*[[paper](https://arxiv.org/abs/2105.01601)]
- [2020/10] **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** *Alexey Dosovitskiy et al. arXiv.*[[paper](https://arxiv.org/abs/2010.11929)]
- [2017/11] **Neural Discrete Representation Learning.** *Aaron van den Oord et al. arXiv.*[[paper](https://arxiv.org/abs/1711.00937)]

#### Audio

- [2023/06] **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding.** *Hang Zhang et al. arXiv.*[[paper](https://arxiv.org/abs/2306.02858)]
- [2023/05] **X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages.** *Feilong Chen et al. arXiv.*[[paper](https://arxiv.org/abs/2305.04160)]
- [2023/05] **InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language.** *Zhaoyang Liu et al. arXiv.*[[paper](https://arxiv.org/abs/2305.05662)]
- [2023/04] **AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head.** *Rongjie Huang et al. arXiv.*[[paper](https://arxiv.org/abs/2304.12995)]
- [2023/03] **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face.** *Yongliang Shen et al. arXiv.*[[paper](https://arxiv.org/abs/2303.17580)]
- [2021/06] **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.** *Wei-Ning Hsu et al. arXiv.*[[paper](https://arxiv.org/abs/2106.07447)]
- [2021/04] **AST: Audio Spectrogram Transformer.** *Yuan Gong et al. arXiv. *[[paper](https://arxiv.org/abs/2104.01778)]

### 1.3 Action: Expand Action Space of LLM-based Agents

#### Tool Using
- [2023/02] Toolformer: Language Models Can Teach Themselves to Use Tools. *Timo Schick et al. arXiv.* [[paper](https://arxiv.org/abs/2302.04761)]
- [2023/04] OpenAGI: When LLM Meets Domain Experts. *Yingqiang Ge et al. arXiv.* [[paper](https://arxiv.org/abs/2304.04370)] [[code](https://github.com/agiresearch/openagi)]
- [2023/05] Large Language Models as Tool Makers. *Tianle Cai et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17126)] [[code](https://github.com/ctlllll/llm-toolmaker)]

#### Embodied Action


## 2. Agents in Practice: Applications of LLM-based Agents

<div align=center><img src="./assets/figure7.jpg" width="60%" /></div>

### 2.1 General Ability of Single Agent
<div align=center><img src="./assets/figure8.jpg" width="60%" /></div>

#### 2.1.1 Task-orietned Deployment
**In web scenarios**
- [2023/07] **WebArena: A Realistic Web Environment for Building Autonomous Agents.** *Shuyan Zhou (CMU) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.13854)] [[code](https://webarena.dev/)]
- [2023/07] **A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis.** *Izzeddin Gur (DeepMind) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.12856)] 
- [2023/06] **SYNAPSE: Leveraging Few-Shot Exemplars for
Human-Level Computer Control.** *Longtao Zheng (Nanyang Technological University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.07863)] [[code](https://github.com/ltzheng/synapse)]
- [2023/06] **Mind2Web: Towards a Generalist Agent for the Web.** *Xiang Deng (The Ohio State University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.06070)] [[code](https://osu-nlp-group.github.io/Mind2Web/)]
- [2023/05] **Multimodal Web Navigation with Instruction-Finetuned Foundation Models.** *Hiroki Furuta (The University of Tokyo) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.11854)] 
- [2023/03] **Language Models can Solve Computer Tasks.** *Geunwoo Kim (University of California) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.17491)] [[code](https://github.com/posgnu/rci-agent)]
- [2022/07] **WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents.** *Shunyu Yao (Princeton University) et al. arXiv.* [[paper](https://arxiv.org/abs/2207.01206)] [[code](https://webshop-pnlp.github.io/)]
- [2021/12] **WebGPT: Browser-assisted question-answering with human feedback.** *Reiichiro Nakano (OpenAI) et al. arXiv.* [[paper](https://arxiv.org/abs/2112.09332)] 

**In life scenarios**
- [2023/08] **InterAct: Exploring the Potentials of ChatGPT as a Cooperative Agent.** *Po-Lin Chen et al. arXiv.* [[paper](https://arxiv.org/abs/2308.01552)] 
- [2023/05] **Plan, Eliminate, and Track -- Language Models are Good Teachers for Embodied Agents.** *Yue Wu (CMU) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.02412)] 
- [2023/05] **Augmenting Autotelic Agents with Large Language Models.** *Cédric Colas (MIT) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.12487)]
- [2023/03] **Planning with Large Language Models via Corrective Re-prompting.** *Shreyas Sundara Raman (Brown University) et al. arXiv.* [[paper](https://arxiv.org/abs/2211.09935)] 
- [2022/10] **Generating Executable Action Plans with Environmentally-Aware Language Models.** *Maitrey Gramopadhye (University of North Carolina at Chapel Hill) et al. arXiv.* [[paper](https://arxiv.org/abs/2210.04964)] [[code](https://github.com/hri-ironlab/scene_aware_language_planner)]
- [2022/01] **Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents.** *Wenlong Huang (UC Berkeley) et al. arXiv.* [[paper](https://arxiv.org/abs/2201.07207)] [[code](https://wenlong.page/language-planner/)]

#### 2.1.2 Innovation-oriented Deployment
- [2023/08] **The Hitchhiker's Guide to Program Analysis: A Journey with Large Language Models.** *Haonan Li (UC Riverside) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.00245)]
- [2023/08] **ChatMOF: An Autonomous AI System for Predicting and Generating Metal-Organic Frameworks.** *Yeonghun Kang (Korea Advanced Institute of Science 
and Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.01423)] 
- [2023/07] **Math Agents: Computational Infrastructure, Mathematical Embedding, and Genomics.** *Melanie Swan (University College London) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.02502)] 
- [2023/06] **Towards Autonomous Testing Agents via Conversational Large Language Models.** *Robert Feldt (Chalmers University of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.05152)] 
- [2023/04] **Emergent autonomous scientific research capabilities of large language models.** *Daniil A. Boiko (CMU) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.05332)] 
- [2023/04] **ChemCrow: Augmenting large-language models with chemistry tools.** *Andres M Bran (Laboratory of Artificial Chemical Intelligence, ISIC, EPFL) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.05376)] [[code](https://github.com/ur-whitelab/chemcrow-public)]
- [2022/03] **ScienceWorld: Is your Agent Smarter than a 5th Grader?.** *Ruoyao Wang (University of Arizona) et al. arXiv.* [[paper](https://arxiv.org/abs/2203.07540)] [[code](https://sciworld.apps.allenai.org/)]

#### 2.1.3 Lifecycle-oriented Deployment
- [2023/05] **Voyager: An Open-Ended Embodied Agent with Large Language Models.** *Guanzhi Wang (NVIDA) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.16291)] [[code](https://voyager.minedojo.org/)]
- [2023/05] **Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory.** *Xizhou Zhu (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17144)] [[code](https://github.com/OpenGVLab/GITM)]
- [2023/03] **Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks.** *Haoqi Yuan (PKU) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.16563)] [[code](https://sites.google.com/view/plan4mc)]
- [2023/02] **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents.** *Zihao Wang (PKU) et al. arXiv.* [[paper](https://arxiv.org/abs/2302.01560)] [[code](https://github.com/CraftJarvis/MC-Planner)]
- [2023/01] **Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling.** *Kolby Nottingham (University of California Irvine, Irvine) et al. arXiv.* [[paper](https://arxiv.org/abs/2301.12050)] [[code](https://deckardagent.github.io/)]

### 2.2 Coordinating Potential of Multiple Agents
<div align=center><img src="./assets/figure9.jpg" width="60%" /></div>

#### 2.2.1 Cooperative Interaction for Complementarity
**Disordered cooperation**
- [2023/07] **Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration.** *Zhenhailong Wang (University of Illinois Urbana-Champaign) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.05300)] [[code](https://github.com/MikeWangWZHL/Solo-Performance-Prompting)]
- [2023/07] **RoCo: Dialectic Multi-Robot Collaboration with Large Language Models.** *Zhao Mandi, Shreeya Jain, Shuran Song (Columbia University) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.04738)] [[code](https://project-roco.github.io/)]
- [2023/04] **ChatLLM Network: More brains, More intelligence.** *Rui Hao (Beijing University of Posts and Telecommunications) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.12998)] 
- [2023/01] **Blind Judgement: Agent-Based Supreme Court Modelling With GPT.** *Sil Hamilton (McGill University). arXiv.* [[paper](https://arxiv.org/abs/2301.05327)] 

**Ordered cooperation**
- [2023/08] **CGMI: Configurable General Multi-Agent Interaction Framework.** *Shi Jinxin (East China Normal University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.12503)]
- [2023/08] **ProAgent: Building Proactive Cooperative AI with Large Language Models.** *Ceyao Zhang (The Chinese University of Hong Kong, Shenzhen) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.11339)] [[code](https://pku-proagent.github.io/)]
- [2023/08] **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents.** *Weize Chen (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.10848)] [[code](https://github.com/OpenBMB/AgentVerse)]
- [2023/08] **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework.** *Qingyun Wu (Pennsylvania State University
) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.08155)] [[code](https://microsoft.github.io/FLAML/docs/Use-Cases/Autogen/)]
- [2023/08] **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework.** *Sirui Hong (DeepWisdom) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.00352)] [[code](https://github.com/geekan/MetaGPT)]
- [2023/07] **Communicative Agents for Software Development.** *Chen Qian (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/openbmb/chatdev)]
- [2023/06] **Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents.** *Yashar Talebira (University of Alberta) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.03314)]
- [2023/05] **Training Socially Aligned Language Models in Simulated Human Society.** *Ruibo Liu (Dartmouth College) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.16960)] [[code](https://github.com/agi-templar/Stable-Alignment)]
- [2023/05] **SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks.** *Bill Yuchen Lin (Allen Institute for Artificial Intelligence) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17390)] [[code](https://yuchenlin.xyz/swiftsage/)]
- [2023/05] **ChatGPT as your Personal Data Scientist.** *Md Mahadi Hassan (Auburn University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.13657)] 
- [2023/03] **CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society.** *Guohao Li (King Abdullah University of Science and Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.17760)] [[code](https://github.com/lightaime/camel)]
- [2023/03] **DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents.** *Varun Nair (Curai Health) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.17071)] [[code](https://github.com/curai/curai-research/tree/main/DERA)]

#### 2.2.2 Adversarial Interaction for Advancement
- [2023/08] **ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate.** *Chi-Min Chan (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.07201)] [[code](https://github.com/thunlp/ChatEval)]
- [2023/05] **Improving Factuality and Reasoning in Language Models through Multiagent Debate.** *Yilun Du (MIT CSAIL) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14325)] [[code](https://composable-models.github.io/llm_debate/)]
- [2023/05] **Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback.** *Yao Fu (University of Edinburgh) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10142)] [[code](https://github.com/FranxYao/GPT-Bargaining)]
- [2023/05] **Examining the Inter-Consistency of Large Language Models: An In-depth Analysis via Debate.** *Kai Xiong (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.11595)] [[code]()]
- [2023/05] **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.** *Tian Liang (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.19118)] [[code](https://github.com/Skytliang/Multi-Agents-Debate)]

### 2.3 Interactive Engagement between Human and Agent
<div align=center><img src="./assets/figure10.jpg" width="60%" /></div>

#### 2.3.1 Instructor-Executor Paradigm

##### Education

- [2023/07] **Math Agents: Computational Infrastructure, Mathematical Embedding, and Genomics.** *Melanie Swan (UCL) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2307.02502)]
  - Communicate with humans to help them understand and use mathematics.
- [2023/03] **Hey Dona! Can you help me with student course registration?.** *Vishesh Kalvakurthi (MSU) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2303.13548)]
  - This is a developed application called Dona that offers virtual voice assistance in student course registration, where humans provide instructions.

##### Health

- [2023/08] **Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue.** *Songhua Yang (ZZU) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2308.03549)] [[code, datasets, and model](https://github.com/SupritYoung/Zhongjing)]
- [2023/05] **HuatuoGPT, towards Taming Language Model to Be a Doctor.** *Hongbo Zhang (CUHK-SZ) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2305.15075)] [[code](https://github.com/FreedomIntelligence/HuatuoGPT)] [[demo](https://www.huatuogpt.cn/)]
- [2023/05] **Helping the Helper: Supporting Peer Counselors via AI-Empowered Practice and Feedback.** *Shang-Ling Hsu (Gatech) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2305.08982)]
- [2020/10] **A Virtual Conversational Agent for Teens with Autism Spectrum Disorder: Experimental Results and Design Lessons.** *Mohammad Rafayet Ali (U of R) et al. IVA '20.* [[paper](https://doi.org/10.1145/3383652.3423900)]

##### Other Application

- [2023/08] **Multi-Turn Dialogue Agent as Sales' Assistant in Telemarketing.** *Wanting Gao (JNU) et al. IEEE.* [[paper](https://doi.org/10.1109/IJCNN54540.2023.10192042)]
- [2023/07] **PEER: A Collaborative Language Model.** *Timo Schick (Meta AI) et al. arXiv.* [[paper](https://openreview.net/pdf?id=KbYevcLjnc)]
- [2023/07] **DIALGEN: Collaborative Human-LM Generated Dialogues for Improved Understanding of Human-Human Conversations.** *Bo-Ru Lu (UW) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2307.07047)]
- [2023/06] **AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn.** *Difei Gao (NUS) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2306.08640)]



#### 2.3.2 Equal Partnership Paradigm

##### Empathetic Communicator

- [2023/08] **SAPIEN: Affective Virtual Agents Powered by Large Language Models.** *Masum Hasan et al. arXiv.* [[paper](xx)] [[code](xx)] [[project page](xx)] [[dataset](xx)]
- [2023/05] **Helping the Helper: Supporting Peer Counselors via AI-Empowered Practice and Feedback.** *Shang-Ling Hsu (Gatech) et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2305.08982)]
- [2022/07] **Artificial empathy in marketing interactions: Bridging the human-AI gap in affective and social customer experience.** *Yuping Liu‑Thompkins et al.* [[paper](https://link.springer.com/article/10.1007/s11747-022-00892-5)]

##### Human-Level Participant

- [2023/08] **Quantifying the Impact of Large Language Models on Collective Opinion Dynamics.** *Chao Li et al. CoRR.* [[paper](https://doi.org/10.48550/arXiv.2308.03313)]
- [2023/06] **Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning.** *Anton Bakhtin et al. ICLR.* [[paper](https://openreview.net/pdf?id=F61FwJTZhb)]
- [2023/06] **Decision-Oriented Dialogue for Human-AI Collaboration.** *Jessy Lin et al. CoRR.* [[paper](https://doi.org/10.48550/arXiv.2305.20076)]
- [2022/11] **Human-level play in the game of Diplomacy by combining language models with strategic reasoning.** *FAIR et al. Science.* [[paper](https://www.science.org/doi/10.1126/science.ade9097)]

## 3. Agent Society: From Individuality to Sociality
<div align=center><img src="./assets/figure12.jpg" width="60%" /></div>

### 3.1 Behavior and Personality of LLM-based Agent

### 3.2 Environment for Agent Society


### 3.3 Society Simulation with LLM-based Agents

- [2023/04] Generative Agents: Interactive Simulacra of Human Behavior. *Joon Sung Park (Stanford) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
   - Agent simulation with LLM-based agents. Some social phenomena are observed.
- [2023/08] AgentSims: An Open-Source Sandbox for Large Language Model Evaluation. *Jiaju Lin (PTA Studio) et al. arXiv.*[[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
   - A simulated environment for evaluation tasks.


## Project Maintainers & Contributors
- Zhiheng Xi （奚志恒, [@WooooDyy](https://github.com/WooooDyy)）


## Contact


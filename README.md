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

### 1.2 Perception: Multimodal Inputs for LLM-based Agents

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

### 2.2 Coordinating Potential of Multiple Agents
<div align=center><img src="./assets/figure9.jpg" width="60%" /></div>

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


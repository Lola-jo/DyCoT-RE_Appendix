## PaperA

这是我的第一批论文，我们称之为 PaperA，已经发表在一篇会议上。

\title{LLM-based Reward Engineering for Reinforcement Learning: A Chain of Thought 
Approach}

\author{
\IEEEauthorblockN{1\textsuperscript{st} Xinning Zhu}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
zhuxinning@shu.edu.cn}
~\\
\and
\IEEEauthorblockN{2\textsuperscript{nd} Jinxin Du}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
jinxin\_du@shu.edu.cn}
~\\
\and
\IEEEauthorblockN{3\textsuperscript{rd} Qiongying Fu}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
fqiongying@163.com}
~\\
\and
\IEEEauthorblockN{4\textsuperscript{th} Lunde Chen*}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
lundechen@shu.edu.cn}
*Corresponding author
}

\maketitle

\begin{abstract}
Reinforcement Learning (RL) has achieved significant milestones in various domains. 
Traditional reward engineering often requires extensive domain knowledge and 
trial-and-error experimentation. This paper explores the integration of Large 
Language Models (LLMs) with CoT reasoning to automate and enhance the generation 
of reward functions for RL environments. We present a comprehensive methodology 
that leverages CoT-enabled LLMs to generate sophisticated, adaptive, and 
context-aware reward functions, implemented within Gymnasium environments and 
trained using Stable Baselines3. Through a detailed case study on the Bipedal 
Walker environment, we demonstrate the efficacy of our approach in producing 
superior agent performance compared to non-CoT approaches.
\end{abstract}

\begin{IEEEkeywords}
Reinforcement Learning, Large Language Models, Chain of Thought, Reward 
Engineering, Gymnasium
\end{IEEEkeywords}

\section{Introduction}

The success of Reinforcement Learning (RL) agents is heavily dependent on the 
design of reward functions that accurately reflect desired behaviors and objectives 
\cite{sutton2018reinforcement}. Traditionally, reward engineering has been a 
manual and iterative process, often requiring deep expertise in the specific 
environment and task \cite{russell2016artificial}. As RL applications become more 
complex, the need for automated and intelligent reward function generation becomes 
paramount.

Large Language Models (LLMs) have shown remarkable capabilities in understanding 
and generating human-like text, which can be harnessed for various applications 
beyond natural language processing \cite{brown2020language}. One such application 
is in the domain of RL reward engineering. By utilizing LLMs with Chain of Thought 
(CoT) reasoning, it is possible to generate reward functions that are not only 
contextually relevant but also logically structured, reducing the reliance on 
manual intervention \cite{baker2019emergent}.

\begin{figure*}[htbp]
\centering
\includegraphics[width=\textwidth]{Figures/cot_structure.jpg}
\caption{Procedure of the Reward Engineering Framework with CoT Integration.}
\label{fig:cot_structure}
\end{figure*}

This paper investigates the use of CoT-enabled LLMs for generating reward 
functions in RL environments. The process begins with an initial system prompt 
that provides a base description of the task and the environment 
(Fig.~\ref{fig:cot_structure}). This prompt includes coding instructions and 
environment code, which serve as the foundation for the LLM to understand the 
context and requirements of the task \cite{ziegler2019fine}.

The CoT integration involves a series of steps where the LLM analyzes the task 
description and environment code carefully. It identifies key objectives and 
constraints, breaks down how different actions should be rewarded, and considers 
potential edge cases. This structured reasoning process ensures that the generated 
reward function is both contextually relevant and logically sound 
\cite{kojima2022large}.

The LLM then iteratively refines the reward function through multiple iterations, 
each guided by the CoT analysis instructions. These instructions include 
performance analysis, reward function analysis, behavioral analysis, and 
improvement planning. The best sample from all iterations is selected and used to 
train the RL agent in the customized environment.

We focus on the Gymnasium framework and employ Stable Baselines3 for training 
agents \cite{towers2024gymnasium}. Through a case study on the Bipedal-Walker 
environment, we illustrate the process and effectiveness of our approach. The 
results demonstrate that CoT-enabled LLMs can significantly improve the efficiency 
and quality of reward function generation, leading to better-performing RL agents.

In summary, this study highlights the potential of integrating CoT reasoning into 
LLMs for automating the reward engineering process in RL. By leveraging the 
structured and logical capabilities of CoT, we aim to reduce the manual effort 
required and enhance the robustness and reliability of RL applications.

\section{Related Work}

\subsection{Reward Function Design}

RL heavily depends on the design of reward functions, which are notoriously 
challenging to construct. Traditional methods have relied 
on domain-specific knowledge and manual crafting, where experts define rewards 
based on task characteristics. However, as tasks grow in complexity, 
these methods become impractical, leading to either sparse or dense rewards that 
lack intermediate guidance or risk overfitting to local optima \cite{mnih2015human}.

Advanced techniques such as Reward Shaping introduce auxiliary rewards to 
facilitate faster convergence without altering the optimal policy \cite{hu2020learning}. 
Potential-Based Reward Shaping (PBRS), proposed by Ng et al., ensures policy 
invariance under certain transformations \cite{ng1999policy}. Inverse Reinforcement Learning (IRL) 
aims to infer reward functions from expert demonstrations, reducing human effort 
in reward design \cite{arora2021survey}. Notable IRL methods include Maximum Entropy IRL \cite{ziebart2008maximum} and 
Guided Cost Learning \cite{finn2016guided}.

Intrinsic motivation methods, like curiosity-driven learning, promote exploration 
by rewarding agents for visiting novel states or acquiring new skills, addressing 
issues associated with sparse rewards \cite{burda2018exploration} \cite{pathak2017curiosity}. These methods enhance the agent's 
ability to explore its environment effectively, particularly when the reward 
signal is sparse or poorly defined.

Recent advancements have seen the integration of LLMs into the process of reward 
function design for RL. This approach leverages the natural language 
understanding capabilities of LLMs to automate the creation of reward functions, 
offering a promising alternative to traditional methods. Kwon et al. \cite{kwon2023reward} 
utilized GPT-3 as a proxy reward function, demonstrating superior performance 
over supervised learning-based rewards even in the absence of examples. Xie et 
al. \cite{xie2023text2reward} introduced the TEXT2REWARD framework, which automatically generates 
dense reward functions from human feedback and achieves comparable results to 
manually crafted rewards in robotic manipulation tasks. Ma et al. \cite{ma2023eureka} developed 
the EUREKA framework, combining environmental context, evolutionary search, and 
reward reflection to produce effective reward functions, especially for complex 
tasks. Song et al. \cite{song2023self} proposed a self-refined LLM framework that designs 
initial reward functions based on natural language input and iteratively refines 
them to align with task requirements.

These studies highlight the potential of LLMs in enhancing RL by automating 
reward design, improving system interpretability, and expanding applicability to 
more complex and varied tasks. The integration of LLMs with RL opens up new 
possibilities for solving intricate real-world problems while maintaining high 
levels of efficiency and reliability.

\subsection{Chain of Thought in LLMs}

Chain-of-Thought (CoT) prompting enhances reasoning capabilities in LLMs, 
addressing limitations in multi-step reasoning tasks despite improvements in model 
size and training data \cite{wei2022chain}. CoT leverages few-shot learning, where models learn 
from examples containing intermediate reasoning steps, promoting structured 
thinking patterns.

Few-shot CoT uses limited demonstrations that include both problem statements and 
corresponding chains of thought, aiding models in breaking down complex problems 
\cite{liu2022few}. Self-consistency generates multiple reasoning paths and selects the most 
consistent answer through voting mechanisms, improving robustness and accuracy 
\cite{wang2022self}. Verifier-based approaches train a component to assess the correctness of 
generated reasoning paths, ensuring higher quality outputs \cite{pan2023automatically}.

Empirical studies demonstrate significant performance improvements in reasoning 
tasks such as arithmetic and commonsense reasoning when applying CoT, 
particularly on benchmarks like MultiArith and GSM8K \cite{wang2024chain}. CoT also increases 
the transparency and interpretability of LLM-generated outputs, facilitating 
broader applications in sophisticated reasoning domains \cite{wu2024usable}.


\section{Methodology}
In this section, we outline the methodology employed in our research to integrate CoT reasoning into the generation of reward functions for RL agents. 

\subsection{Prompt Design for CoT-Enabled Reward Generation}

The initial phase involves using CoT reasoning, guided by a LLM, to generate structured reward function components \( r_i(s, a) \), 
which are dependent on states \( s \) and actions \( a \). Each component originates from specific input information \( I_i \):

\begin{equation}
r_i(s, a) = \text{CoT}(I_i)
\label{eq:cot_reward}
\end{equation}

Simultaneously, LLM assigns an initial weight \( w_i \) to each component, establishing the base reward function:

\begin{equation}
R_{base} = \sum_{i=1}^{m} w_i \cdot r_i(s, a)
\label{eq:base_reward}
\end{equation}

This ensures that each reward function component is underpinned by coherent thought processes and logically contributes to achieving the task objectives.

\subsection{Leveraging CoT in Reward Function Generation}

The RL agent is trained within the customized Gymnasium environment using Stable Baselines3 (SB3). The training process follows the policy gradient update 
rule, incorporating the CoT-generated reward function to guide the learning process:

\begin{equation}
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta; \mathbf{s}, \mathbf{a})
\label{eq:policy_update}
\end{equation}

where \(\theta\) denotes the policy parameters, \(\alpha\) is the learning rate, and \(J(\theta; \mathbf{s}, \mathbf{a})\) represents the objective function over 
states \(\mathbf{s}\) and actions \(\mathbf{a}\).

The performance of each reward function component during training is evaluated using fitness scores \( FS(r_i^{(k)}) \). The CoT reasoning process extracts logical 
reasoning and structured reward components from the LLM's response, which can then be iteratively improved based on observed performance metrics and domain knowledge. 
Subsequently, through CoT reflection, adjustments are made to the reward function components and their weights for the next iteration:

\begin{equation}
R_{new}^{(k)} = \text{CoT\_Reflection}(FS(r_i^{(k)}), W^{(k)})
\label{eq:reflection}
\end{equation}

where \( W^{(k)} \) represents the set of weights at iteration \( k \).

\subsection{Final Evaluation and Analysis}

Throughout these iterations, the best-performing reward function configuration and its corresponding weights are tracked and updated:

\begin{equation}
\text{best\_config}^{(k)} = \arg\max_{j \leq k} \{FS(r_i^{(j)}, W^{(j)})\}
\label{eq:best_config}
\end{equation}

Upon completing all \( n \) iterations, a final evaluation is performed on the overall best sample identified across all iterations. This assessment involves 
re-evaluating the top-performing sample with different random seeds to ensure robustness and generalizability. By doing so, we gain deeper insights into the 
stability and reliability of the reward function and its components, ensuring that the selected sample can perform well under varying conditions.

\subsection{Summary}

By continuously evaluating and adjusting the components and weights of reward functions using CoT reasoning, 
this method enhances the precision with which reward functions reflect task goals. It promotes more efficient learning 
and improved performance of RL agents. The iterative refinement process, combined with final evaluation, ensures the 
optimization of reward function configurations while maintaining clarity and effectiveness throughout the training process. 
CoT not only guides the generation and adjustment of reward functions but also ensures that the reasoning behind these adjustments 
remains transparent and logical, thereby enhancing the overall quality of the RL agent's training.

\section{Experimental Procedure}
This section details the implementation of our methodology in the Bipedal Walker environment, 
following the theoretical framework established in Section III. We present the experimental 
protocol that demonstrates how CoT reasoning guides the iterative improvement of reward functions. 
The full appendix is available at GitHub\footnote{\url{https://github.com/evidentiallab/RewardEngineering_CoT/blob/main/appendix.pdf}}.

\subsection{Experimental Setup and Configuration}

The experiments were conducted using Gymnasium's BipedalWalker-v3 environment, with the PPO algorithm 
implemented through Stable-Baselines3 serving as the primary training method. Each training iteration 
consisted of 1 million steps to ensure sufficient learning opportunity. The computational infrastructure 
comprised two NVIDIA GeForce RTX 3090 GPUs with 24GB memory each, while Llama-3.1 was employed for CoT reasoning processes.

To evaluate the performance, we established multiple assessment criteria aligned with the theoretical 
framework presented in Section III. The primary evaluation metrics include the fitness score 
reflecting the agent's performance, episode length indicating task completion efficiency.

\subsection{CoT-Guided Reward Function Design}

The reward function design process follows three main phases, implementing the theoretical framework
 developed in Section III. The process begins with an initial design phase, followed by iterative refinement, and concludes with environment integration.

\subsubsection{Initial CoT-Based Design Phase}
The initial phase commences with a task analysis, where we decompose the Bipedal Walker objectives 
into fundamental components. Following \eqref{eq:cot_reward}, we derive the initial reward components \(r_i(s, a)\) 
through CoT reasoning, as shown in Fig.~\ref{fig:cot_reward_design}. These components are then assigned preliminary weights according to \eqref{eq:base_reward}, 
establishing the foundation for subsequent refinement.

\begin{figure}[htbp]
    \centerline{\includegraphics[width=0.9\columnwidth]{Figures/design-reward.png}}
    \caption{CoT-Based Reward Function Design Process.}
    \label{fig:cot_reward_design}
    \end{figure}

The refinement process implements \eqref{eq:policy_update}, encompassing a systematic analysis of
agent behavior and performance metrics. 

\subsubsection{CoT-Guided Iterative Refinement Phase}
The refinement process implements \eqref{eq:reflection}, encompassing a systematic analysis of
 agent behavior and performance metrics, as shown in Fig.~\ref{fig:cot_reward_reflection}. Through this analysis, 
 we continuously adjust both the reward components \(r_i(s, a)\) and their associated weights \(W_i\). 
 This iterative process ensures that the reward function evolves to better align with desired behaviors.

\begin{figure}[htbp]
\centerline{\includegraphics[width=0.9\columnwidth]{Figures/reward-reflection.png}}
\caption{CoT-Based Reward Function Reflection and Refinement Process.}
\label{fig:cot_reward_reflection}
\end{figure}


\subsection{Training and Evaluation Protocol}

The evaluation phase involves integrating the refined reward function into the Gymnasium environment. 
This step requires implementing a custom environment wrapper to
 apply the new reward function, along with normalization and scaling procedures. The modified environment 
 is then integrated into the PPO training pipeline to ensure consistent and reliable training processes.

We begin by establishing a benchmark through training the agent using the default reward function and 
recording performance metrics. This baseline serves as a reference for subsequent comparisons.

In iterations 0 through n, we implement the CoT-based refined 
reward function. Each iteration generates a new reward function variant used to train the agent. 
Performance data across all relevant metrics are collected during these runs. 

Once the best-performing sample is identified from the iterative process, it undergoes re-evaluation 
with different random seeds. This ensures that the robustness of the reward function is assessed across
 various initial conditions, providing a more comprehensive understanding of its stability and reliability.

The outcomes of this experimental procedure are presented in Section V, where we provide an in-depth analysis of our CoT-based approach's effectiveness. 

\section{Results of Experiment}

In our experiments, we use the default reward function implementation from Gymnasium's BipedalWalker-v3 environment, coupled with Stable-Baselines3's PPO algorithm, as our benchmark method. This standard implementation is widely recognized in the RL community and serves as a reliable baseline for comparison. For fairness, both our CoT-based approach and the benchmark use identical hyperparameters. Detailed settings are provided in Appendix B.2.
\subsection{Using CoT to start from scratch}

The experiment utilizing Chain of Thought (CoT) reasoning to design a reward function from scratch in the Bipedal Walker environment demonstrated significant improvements in the agent's performance metrics. The fitness score, an indicator of the agent's proficiency within the environment, was carefully monitored over several iterations.

\begin{figure}[htbp]
\centerline{\includegraphics[width=0.9\columnwidth]{Figures/figure-cot-from-scratch.png}}
\caption{Fitness Score Per Episode.}
\label{fig:fitness_scratch}
\end{figure}

As illustrated in Fig.~\ref{fig:fitness_scratch}, the fitness score increased rapidly during the early iterations. This sharp rise indicates that the agent effectively acquired the essential skills for navigating the environment, ultimately reaching a performance plateau at approximately 300 points. This plateau suggests that the agent's proficiency stabilized after the learning phase.

In summary, the application of CoT reasoning in reward function design facilitated rapid learning and stable performance in the Bipedal Walker environment. The evolution of the fitness score over iterations highlights the method's effectiveness in guiding the agent towards optimal behaviors.

\subsection{Comparison with Benchmark Method}

In this section, we compare the performance of the CoT-based reward function against the benchmark method (Gymnasium's default reward function with Stable-Baselines3's PPO) across various metrics. The performance metrics considered include fitness score and episode length, evaluated across different batch sizes. 

As shown in Table \ref{tab:performance_comparison_fitness}, the CoT method consistently yields higher mean fitness scores and exhibits lower standard deviations, indicating superior and more stable performance compared to the benchmark. Additionally, the CoT method achieves higher maximum fitness scores and demonstrates significantly better smoothness, reflecting more consistent agent behavior across episodes.

\begin{table}[htbp]
    \caption{Fitness Score Comparison Across Batch Sizes}
    \begin{center}
    \begin{tabular}{|l|c|c|c|c|}
    \hline
    \multirow{2}{*}{\textbf{Metric}} & \multicolumn{2}{c|}{\textbf{Batch=32}} & \multicolumn{2}{c|}{\textbf{Batch=64}} \\
    \cline{2-5}
    & \textit{CoT} & \textit{Benchmark} & \textit{CoT} & \textit{Benchmark} \\
    \hline
    Mean & 274.15 & 245.31 & 248.49 & 222.05 \\
    Std  & 82.51 & 105.41 & 83.08 & 115.27 \\
    Max  & 304.28 & 298.82 & 301.20 & 295.95 \\
    Min  & -106.24 & -109.34 & -94.28 & -121.44 \\
    Smoothness & 6.88 & 38.90 & 21.70 & 47.42 \\
    \hline
    \end{tabular}
    \label{tab:performance_comparison_fitness}
    \end{center}
\end{table}

Table \ref{tab:performance_comparison_episode} highlights that the CoT method leads to faster convergence, as indicated by shorter episode lengths compared to the benchmark. This suggests that the CoT-based reward function enables the agent to reach the goal more quickly, facilitating faster learning and more efficient task completion.

\begin{table}[htbp]
    \caption{Episode Length Comparison Across Batch Sizes}
    \begin{center}
    \begin{tabular}{|l|c|c|c|c|}
    \hline
    \multirow{2}{*}{\textbf{Metric}} & \multicolumn{2}{c|}{\textbf{Batch=32}} & \multicolumn{2}{c|}{\textbf{Batch=64}} \\
    \cline{2-5}
    & \textit{CoT} & \textit{Benchmark} & \textit{CoT} & \textit{Benchmark} \\
    \hline
    Mean & 1060.06 & 1068.24 & 1016.56 & 1186.05 \\
    Std  & 183.93 & 286.21 & 186.25 & 330.26 \\
    Max  & 1321.50 & 1600.00 & 1178.00 & 1600.00 \\
    Min  & 123.88 & 47.13 & 95.88 & 40.00 \\
    Smoothness & 33.66 & 140.83 & 74.96 & 177.36 \\
    \hline
    \end{tabular}
    \label{tab:performance_comparison_episode}
    \end{center}
\end{table}

These results underscore the CoT-based reward function's effectiveness in improving both agent performance and learning efficiency, surpassing the benchmark in stability, convergence speed, and task completion.
\subsection{The effectiveness of CoT}

In this section, we delve into the comparative analysis of the detailed CoT approach against the non-CoT method in the context of optimization search processes. 

The experimental results highlight the improvements when incorporating CoT reasoning into the reward engineering process. As illustrated in Fig.~\ref{fig:convergence_comparison},
the CoT approach exhibits rapid convergence with fewer iterations required to reach high fitness values. Data points for the CoT method are predominantly located in the upper-left region 
of the graph, indicating efficient convergence to superior solutions within a shorter timeframe compared to the non-CoT method, which shows more scattered data points and requires more 
iterations to achieve similar or lower fitness levels.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\columnwidth]{Figures/best-converge.png}
\caption{Comparison of Convergence Speed vs Best Fitness between non-CoT (left) and CoT (right) approaches.}
\label{fig:convergence_comparison}
\end{figure}


In addition, Fig.~\ref{fig:stability_comparison} shows that the data points using the CoT method are mainly clustered in the upper left part, indicating that the algorithm can also achieve higher final performance under the condition of low standard deviation (i.e. high stability). In contrast, non CoT methods have a wider range of data points, especially in the high standard deviation range. This shows that with the guidance of CoT thinking chain, the fluctuation of the algorithm is reduced, and the final performance results are more consistent and reliable.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\columnwidth]{Figures/final-stablity.png}
\caption{Comparison of Stability vs Final Performance between non-CoT (left) and CoT (right) approaches.}
\label{fig:stability_comparison}
\end{figure}


The data presented in Table~\ref{tab:comparison} further substantiates the aforementioned observations. The mean fitness values are consistently higher across all iterations with CoT, indicating more effective optimization. The common range, which shows the interval with the highest frequency of fitness values, is predominantly positive for CoT, contrasting with the negative ranges often seen in non-CoT methods. This highlights CoT's role in achieving better and more stable performance outcomes.

\begin{table}[h!]
\centering
\caption{Comparison of non-CoT and CoT Results Over Iterations.}
\begin{tabular}{|c|c|c|c|c|}
\hline
Iteration & \multicolumn{2}{c|}{Mean Fitness} & \multicolumn{2}{c|}{Common Range} \\
\cline{2-5}
& non-CoT & CoT & non-CoT & CoT \\
\hline
0 & 1.38    & 122.34 & [-92.1, -14.2]   & [217.5, 309.9] \\
1 & 58.61   & 188.20 & [-91.7, -13.7]   & [226.9, 309.8] \\
2 & 67.17   & 202.37 & [-106.9, -24.6]  & [226.4, 309.8] \\
3 & 87.05   & 201.39 & [-92.1, -14.5]   & [237.8, 312.0] \\
4 & 66.11   & 206.04 & [-138.3, -50.0]  & [241.3, 307.5] \\
5 & 83.38   & 169.15 & [222.6, 303.2]   & [242.1, 309.9] \\
6 & 97.31   & 179.22 & [221.3, 303.2]   & [230.1, 310.6] \\
7 & 74.34   & 159.73 & [-91.5, -14.8]   & [239.3, 310.6] \\
\hline
\end{tabular}
\label{tab:comparison}
\end{table}

Overall, the experimental results indicate that the incorporation of CoT reasoning provides benefits in terms of acceleration of convergence, 
enhancement of performance, and improvement of result stability. These enhancements contribute to the development of more robust and efficient RL algorithms.

\section{Conclusion}

To conclude, this study demonstrates the efficacy of Chain of Thought (CoT)-enabled Large Language Models in generating reward functions for Reinforcement Learning (RL) environments. The proposed method, as illustrated through a detailed case study on the Bipedal-Walker environment, significantly enhances the quality of reward structures, leading to improved agent performance. Ablation studies further validate the contribution of CoT reasoning in refining reward function design. While the approach shows substantial promise, its applicability across diverse RL domains requires further exploration. Future work should focus on evaluating its effectiveness in more complex environments and addressing potential limitations to facilitate broader adoption in RL applications
\end{document}     

以上为 PaperA的 latex 内容。

## PaperB

这是我正在准备的第二篇论文，但是我们需要好好讨论，我们称之为 PaperB: 
\title{Dual-Dynamic Optimization for RL Reward Functions: \\ Synergistic Temperature Regulation and Model Selection}


\author{
\IEEEauthorblockN{1\textsuperscript{st} Xinning Zhu}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
zhuxinning@shu.edu.cn}
~\\
\and
\IEEEauthorblockN{2\textsuperscript{nd} Jinxin Du}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
jinxin\_du@shu.edu.cn}
~\\
\and
\IEEEauthorblockN{3\textsuperscript{rd} Qiongying Fu}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
fqiongying@163.com}
~\\
\and
\IEEEauthorblockN{4\textsuperscript{th} Lunde Chen*}
\IEEEauthorblockA{\textit{Sino-European School of Technology} \\
\textit{Shanghai University}\\
Shanghai, China \\
lundechen@shu.edu.cn}
*Corresponding author
}

\maketitle

\begin{abstract}
Chain-of-Thought (CoT) reasoning methods hold great potential in automating reward function design for reinforcement learning (RL), especially when combined with large language models (LLMs). However, existing CoT-based frameworks often rely on static configurations, limiting their adaptability in dynamic or complex environments. This paper presents an adaptive CoT reward generation framework that incorporates two optimization mechanisms: Dynamic Temperature Regulation via Optimization (DTRO) and Dynamic Model Selection for Reward Optimization (DMSRO). The proposed system demonstrates superior adaptability, learning stability, and sample efficiency across various standard and custom RL environments.
\end{abstract}

\begin{IEEEkeywords}
Reinforcement learning, reward engineering, large language models, chain-of-thought reasoning, dynamic temperature adjustment, model selection
\end{IEEEkeywords}

\section{Introduction}
Reinforcement learning (RL) has achieved remarkable success in domains such as game playing \cite{mnih2015human}, robotic control \cite{finn2016guided}, and collaborative behavior modeling \cite{baker2019emergent}. Yet, the design of effective and generalizable reward functions remains a fundamental challenge, often relying on manual engineering \cite{ng1999policy, arora2021survey}.

The emergence of large language models (LLMs) \cite{brown2020language, ouyang2022training, achiam2023gpt} and their reasoning capabilities, particularly through Chain-of-Thought (CoT) prompting \cite{kojima2022large, dua2022, wang2023e}, has opened new opportunities for reward function automation. Nevertheless, most CoT-based methods adopt fixed model configurations and static sampling parameters, which hampers their adaptability in evolving RL environments.

In this work, we address this gap by proposing an adaptive optimization framework that enhances CoT-based reward generation with two dynamic mechanisms: temperature regulation (DTRO) and model selection (DMSRO). This approach enables runtime adaptability and better exploration-exploitation tradeoffs.

\section{Related Work}
Traditional reward engineering methods, including reward shaping \cite{sutton1998reinforcement, hu2020learning} and inverse reinforcement learning \cite{ziebart2008maximum}, offer theoretical foundations but lack scalability. Recent LLM-based systems such as EUREKA \cite{ma2023eureka} and Text2Reward \cite{xie2023text2reward} leverage natural language but typically operate under static configurations. Chain-of-Thought prompting enhances reasoning capabilities \cite{wang2023c, madaan2023}, while temperature control \cite{zhu2024hot, cecere2025monte, zhang2024edt} and model adaptation \cite{hsieh2023distilling, vardhni2024performance} remain underexplored in reward generation.



\section{Methodology}
\subsection{Architecture Overview}
The proposed framework combines evolutionary search with dynamic reward optimization, as illustrated in Fig.~\ref{fig:architecture} and Fig.~\ref{fig:evolution}. The system processes natural language inputs (e.g., "Design a reward function for stable bipedal robot walking") through a dual-path mechanism:

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{./Figures/architecture.png}
\caption{Reward function design framework. Top: Natural language task specification and environment interface. Middle: Chain-of-Thought reasoning (left) and dynamic optimization modules (right). Bottom: Executable output and monitoring system.}
\label{fig:architecture}
\end{figure}

\subsection{Chain-of-Thought Generation}
The left branch in Fig.~\ref{fig:architecture} demonstrates the three-stage decomposition process:
\begin{equation}
\text{CoT}(d) \rightarrow \begin{cases}
\text{Decompose goal} & \text{(e.g., balance, forward motion)} \\
\text{Identify key states} & \text{(e.g., torso angle $\theta$)} \\
\text{Mathematical modeling} & \text{(e.g., $r_t = w_1\cos\theta + w_2v_x$)}
\end{cases}
\end{equation}

\subsection{Dynamic Optimization}
The evolutionary loop in Fig.~\ref{fig:evolution} operates through:
\begin{equation}
P_{t+1} = \underbrace{\text{Select}(P_t, k=0.2)}_{\text{Elite selection}} \oplus \underbrace{\text{Mutate}(\theta, T)}_{\substack{\text{Temperature-}\\ \text{controlled}}}
\end{equation}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{./Figures/evolution.png}
\caption{Evolutionary search loop for reward optimization. The process iteratively refines reward functions through parallel RL training, with fitness evaluation driving selection and mutation.}
\label{fig:evolution}
\end{figure}

The DTRO module regulates exploration-exploitation balance via entropy-aware temperature adjustment:
\begin{equation}
\Delta T = \beta \frac{\partial H}{\partial t} \cdot \mathbb{I}(\sigma_R > \tau)
\end{equation}

where $\mathbb{I}(\cdot)$ is the indicator function. The DMSRO component, shown in the right branch of Fig.~\ref{fig:architecture}, optimizes model selection through:
\begin{equation}
m^* = \mathop{\mathrm{arg\,max}}\limits_{m \in \mathcal{M}} \left( \alpha \cdot \text{Score}(m) + (1-\alpha)\cdot \text{Efficiency}(m) \right)
\end{equation}

\subsection{Integration Mechanism}
The two diagrams collectively demonstrate how initial CoT-generated rewards evolve through:
\begin{itemize}
\item Continuous refinement via the evolutionary loop (Fig.~\ref{fig:evolution})
\item Real-time adaptation through dynamic modules (Fig.~\ref{fig:architecture})
\end{itemize}

The fitness function $F=0.4S+0.3C+0.3E$ in Fig.~\ref{fig:evolution} ensures balanced optimization of stability ($S$), completion ($C$), and efficiency ($E$).

The proposed framework models reward generation as a function of task description, temperature, and model:
\begin{equation}
R(s,a,t) = \Phi(d, T(t), m(t))
\end{equation}
where $\Phi$ denotes the CoT-based LLM generation process.

\subsection{Dynamic Temperature Regulation (DTRO)}
DTRO regulates the LLM sampling temperature based on policy entropy $H_t$ and confidence $C_t$. The update rule is:
\begin{equation}
\Delta T_t = \beta \Delta T_{t-1} + (1-\beta) \left[\alpha_1 \tanh\left(\frac{H_t - \bar{H}}{\sigma_H}\right) + \alpha_2(C_t - \theta_c)\right]
\end{equation}
This mechanism adapts the creativity and determinism of LLM outputs according to policy performance.

\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.95\textwidth]{./Figures/reward_episode_curve.png}
    \caption{Average reward over training episodes in five environments. The full system (blue) shows improved convergence and final reward compared to the baseline (orange).}
    \label{fig:reward-episode}
\end{figure*}

\subsection{Dynamic Model Selection (DMSRO)}
Let $\mathcal{M}$ be a set of candidate models. The selection score for model $m$ is:
\begin{equation}
p_{\text{fused}}(m) = (1-\gamma) \cdot p_{\text{local}}(m) + \gamma \cdot p_{\text{hist}}(m)
\end{equation}
An $\epsilon$-greedy strategy is applied to explore new models while exploiting high-performing ones.

\section{Experiments}
\label{sec:exp}



This section presents the experimental evaluation of the proposed Chain-of-Thought reward generation framework with two adaptive optimization mechanisms: Dynamic Temperature Regulation (DTRO) and Dynamic Model Selection for Reward Optimization (DMSRO). Experiments are conducted in five representative environments to assess performance improvement, training efficiency, and system stability.

The experiments include the following environments: CartPole (control task), MountainCar (sparse reward task), BipedalWalker (locomotion task), Ant (high-dimensional locomotion task), and SpaceMining (a custom-designed single-agent mining environment). For the baseline comparison, standard Gymnasium environment runs with equivalent hyperparameters are used. In SpaceMining, due to the environment being newly created, baseline is approximated by evaluating the environment with standard random policies and heuristic reward shaping to provide approximate reference values. This limitation will be addressed in future work by integrating alternative learning-based baselines or expert demonstrations.

\subsection{Overall Reward Performance}

Figure \ref{fig:reward_curve} shows the average reward curves over training episodes for the full system compared to the baseline in each environment. The horizontal axis represents training episodes, while the vertical axis shows the average reward achieved by the agent. Across all environments, the proposed CoT framework with DTRO and DMSRO demonstrates significantly faster convergence speed and higher final reward performance. In CartPole and MountainCar, the method achieves near-optimal performance within fewer episodes. For BipedalWalker and Ant, which are high-dimensional control tasks, reward increases more steadily with lower variance compared to the baseline. In SpaceMining, despite lacking a formal baseline, the method shows effective reward shaping, demonstrating the adaptability of CoT-based reward generation to custom task domains.



Table~\ref{tab:performance-comparison} presents the quantitative results, reporting average reward, maximum reward, standard deviation, and average convergence episodes. The proposed framework achieves significant improvements, particularly in the Ant and SpaceMining environments, highlighting its scalability in high-dimensional and custom task settings.

\begin{table}[ht]
\caption{Performance comparison across environments}
\label{tab:performance-comparison}
\centering
\begin{tabular}{lcccc}
\hline
Environment & Avg. Reward & Max Reward & Std. Dev & Conver. Ep. \\
\hline
CartPole & 195.2 & 200.0 & 4.3 & 110 \\
MountainCar & -110.4 & -85.2 & 12.8 & 350 \\
BipedalWalker & 312.4 & 340.1 & 15.5 & 420 \\
Ant & 2867.5 & 3150.0 & 185.7 & 920 \\
SpaceMining & 218.7 & 240.3 & 21.1 & 1350 \\
\hline
\end{tabular}
\end{table}

\subsection{Temperature-Entropy-Reward Correlation Analysis}

Figure \ref{fig:temp_entropy_reward} illustrates the three-dimensional heatmap of temperature, policy entropy, and average reward under the DTRO mechanism. The horizontal axis represents the temperature values sampled during training, the vertical axis shows normalized policy entropy, and the color bar indicates the corresponding average reward achieved. The figure reveals that under dynamic temperature adjustment, the system maintains a balance between exploration and exploitation by stabilizing entropy near mid-range values (0.4-0.6) while progressively lowering temperature as the policy converges. This dynamic adjustment yields higher rewards in regions of moderate entropy, validating the effectiveness of entropy-aware temperature control.

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.45\textwidth]{./Figures/temp_entropy_reward_heatmap.png}
  \caption{Temperature-entropy-reward correlation heatmap under DTRO. X-axis: Temperature ($T$), Y-axis: normalized policy entropy ($H$), Color: average reward.}
  \label{fig:temp_entropy_reward}
\end{figure}

Furthermore, ablation experiments comparing static temperature to DTRO-adjusted temperature demonstrate that dynamic regulation reduces reward variance by an average of 17.2\% and improves convergence speed by 13.5\%.

\subsection{Dynamic Model Selection Analysis}

Figure \ref{fig:model_switch_log} presents the model switching log visualization under the DMSRO mechanism. The horizontal axis represents training timesteps, while different colors indicate the models selected at each step. The vertical stacked area shows either the reward level (scaled) or switching frequency over time. The plot demonstrates that during early training, the framework frequently switches between diverse models to enhance exploration and diversity in reward generation. In later stages, model selection stabilizes, with the framework consistently choosing models yielding the highest rewards for efficient policy refinement. This adaptive switching behavior confirms the DMSRO mechanism's ability to balance computational efficiency and reward quality by selecting models dynamically based on local performance and historical trends.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.45\textwidth]{./Figures/model_switch_log.png}
    \caption{DMSRO model switching log. X-axis: timestep, color: selected model (LLaMA-3, Qwen-2.5, DeepSeek-R1), line: reward progression. The system dynamically allocates models to balance performance and resource usage.}
    \label{fig:model-switch-log}
\end{figure}

Table~\ref{tab:dmsro-performance} summarizes the reward performance under different model selection strategies. DMSRO achieves an optimal balance between performance and GPU resource consumption, reducing computational hours by approximately 14.8\% while maintaining superior reward levels.

\subsection{Joint System Performance}

Table \ref{tab:overall_perf} summarizes the joint system performance across environments. Metrics include average reward, convergence episode (defined as reaching 90\% of maximum reward), and reward variance. Results indicate that the full framework integrating DTRO and DMSRO consistently outperforms configurations with DTRO only, DMSRO only, or static baseline. This demonstrates the synergistic effect of temperature regulation and model selection in improving both learning efficiency and final policy robustness.

\begin{table}[ht]
\centering
\caption{Joint system performance comparison across environments}
\label{tab:overall_perf}
\begin{tabular}{l|cccc}
\hline
\textbf{Env} & \textbf{Reward (↑)} & \textbf{Conv.(↓)} & \textbf{Var. (↓)} & \textbf{Config} \\
\hline
CartPole & 210.7 & 75 & 8.5 & DTRO + DMSRO \\
MountainCar & 93.5 & 140 & 12.3 & DTRO + DMSRO \\
BipedalWalker & 312.4 & 480 & 38.6 & DTRO + DMSRO \\
Ant & 2750.9 & 980 & 201.4 & DTRO + DMSRO \\
SpaceMining & 134.2 & 560 & 45.8 & DTRO + DMSRO \\
\hline
\end{tabular}
\end{table}

Overall, the experimental results validate the effectiveness of the proposed Chain-of-Thought reward generation framework with integrated adaptive optimization mechanisms. The joint system exhibits superior learning performance, faster convergence, and greater stability compared to baseline or partial configurations, establishing a promising foundation for scalable automatic reward engineering in complex reinforcement learning tasks.

Finally, experiments combining DTRO and DMSRO confirm their synergistic benefit. Figure~\ref{fig:dual-mechanism} illustrates the comparative performance of four system configurations: baseline, DTRO only, DMSRO only, and the full system. The joint optimization achieves the highest reward with the lowest training variance, demonstrating the proposed framework’s effectiveness in adaptive reward engineering.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.45\textwidth]{./Figures/dual_mechanism_comparison.png}
    \caption{Performance comparison across system configurations. Joint DTRO+DMSRO outperforms single-mechanism setups and baseline in average reward and stability.}
    \label{fig:dual-mechanism}
\end{figure}

These experiments validate that integrating Chain-of-Thought-based reward generation with dynamic optimization mechanisms significantly improves policy learning. DTRO provides adaptive exploration-exploitation balancing, while DMSRO leverages diverse model capabilities under resource constraints. Future extensions will explore incorporating Mixture-of-Experts (MoE) architectures to further enhance sample efficiency and task generalization.




% \begin{figure}[!ht]
% \centering
% \includegraphics[width=0.9\linewidth]{temperature_correlation.pdf}
% \caption{Reward–temperature correlation in CartPole (DTRO active)}
% \label{fig:temp_corr}
% \end{figure}

\section{Discussion}
While our method improves reward adaptability and task generalization, limitations persist in model switching costs and reward interpretability. Future efforts may include structured prompting \cite{chen2022, gao2023}, hybrid reward learning \cite{skalse2022defining}, and MoE architecture integration.

\section{Conclusion}
We propose a dual-dynamic optimization framework for CoT-based RL reward generation, incorporating temperature regulation and model selection. Our results demonstrate improved convergence, stability, and reward quality across tasks, laying a foundation for scalable, self-adaptive reward engineering. 

以上为 PaperB 的内容。
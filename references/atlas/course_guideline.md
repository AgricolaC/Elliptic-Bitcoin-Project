**Lecturers**: [Vaccarino Francesco](https://didattica.polito.it/pls/portal30/sviluppo.scheda_pers_swas.show?m=11485)
**Co-lecturers:** [Gasparini Mauro](https://didattica.polito.it/pls/portal30/sviluppo.scheda_pers_swas.show?m=2692)

This advanced course explores the convergence of geometric learning, graph neural networks (GNNs), and time-variant data analysis, with a strong emphasis on anomaly detection in complex and dynamic systems. The course begins with the foundations of geometric machine learning, focusing on the representation of data in non-Euclidean domains, such as graphs and manifolds. It then progresses to cutting-edge developments in graph neural networks, including architectures adapted to temporal and heterogeneous graph data. The course also introduces techniques for time-series modeling, both classical and modern (e.g., machine learning and deep learning approaches), and culminates in a comprehensive module on anomaly detection, covering both statistical and learning-based methods. Throughout the course, students will engage with real-world datasets and hands-on programming exercises, gaining both theoretical insights and practical expertise.

**Course Topics:**
###### Module 1: Foundations of Geometric Learning
_Focus: Representations in non-Euclidean domains, manifolds, and symmetries._
- [[Group Theory in Data: Symmetries & Equivariance]]
- [[Manifold Learning - Isomap, LLE, and Diffusion Maps]]
- [[The Geometry of Hilbert Spaces]] (Linked from MML)
###### Module 2: Graph Neural Networks (GNNs)
_Focus: Deep learning on graph-structured data._
- [[Message Passing Neural Networks (MPNN)]]
- [[GNN Architectures: GCN, GAT, and GIN]]
- [[Temporal GNNs: Modeling Dynamic and Heterogeneous Graphs]]
###### Module 3: Time-Variant Data & Time-Series Modeling
_Focus: Classical and Deep Learning approaches to temporal sequences._
- [[Classical Time-Series]]
- [[State-Space Models and Kalman Filters]]
- [[Deep Sequential Learning]]
	- RNN
	- LSTM
	- TCN
	- Temporal GNN
- [[Transformers for Temporal Data]]
###### Module 4: Anomaly Detection in Complex Systems
_Focus: Statistical and learning-based detection in time and graph data._
- [[Statistical Anomaly Detection]]
- [[Machine Learning for Outliers]]
	- Isolation Forest
	- One-class SVM
- [[Reconstruction-Based Detection]]
- [[Generative Adversarial Networks]]
- [[Graph-Based Anomaly Detection]]
	- Spectral Methods
	- Subgraph Detection
###### Project: [[Elliptic Bitcoin Dataset]]

 Class materials, datasets, and code notebooks will be made available on the course GitHub repository. • Guest lectures and seminars by experts in AI and complex systems may be included. • The course can be extended to include a mini-research project for PhD students.

**Reading Materials**: 
- Geometric Learning Bronstein et al., Geometric Deep Learning (2024) 
- Graph Neural Networks Hamilton, Graph Representation Learning (2020) 
- Time Series Analysis Hyndman & Athanasopoulos, 
- Anomaly Detection Aggarwal, Outlier Analysis (2017) 
- Applied Time Series Analysis: A Practical Guide to Modeling and Forecasting (Mills)

Student practices with exercises from the books, past exams, and in-class exercises, in addition to reading the books and studying the concepts.

**_Assessment and grading criteria_**

**Exam:** Individual essay;

Mini project report describing their workflow, rationale, and findings. (0-16 points) Oral discussion of their conclusions and questions on the theoretical aspects of the methodologies they implied in their project. (0-16 points) Weight: 50% mini project/50% oral exam. The laude is assigned to students with a total score higher than 30 points.

**_Reference Material_**

All reference material is provided by the professor to guide students in the composition of their final project.

- **`references/notebooks/`** — Lab notebooks covering the course modules. These serve as worked examples and building blocks for the final project:
  - `nb1_foundations.ipynb` — Foundations of deep learning
  - `nb1 symmetries exec.ipynb` — Symmetries in neural networks
  - `nb2_timeseries_dl.ipynb` — Time series analysis with deep learning
  - `nb2 invariance equivariance exec.ipynb` — Invariance and equivariance exercises
  - `nb3_autoencoders.ipynb` — Autoencoders
  - `nb3b_intrinsic_dim_tsne_umap.ipynb` — Intrinsic dimensionality, t-SNE, and UMAP
  - `lecture3 dominant.ipynb` — Dominant eigenvalue analysis
  - `lstm_lecture.ipynb`, `lstm_ae_notebook.ipynb`, `nb4_lstm_ae.ipynb` — LSTM and LSTM-Autoencoder architectures
  - `nb4 gauge theories exec.ipynb` — Gauge theories exercises
  - `nb5_gans_graphs.ipynb` — GANs and graph neural networks
  - `student_exam_elaborato_ecg5000_amended.ipynb` — Example student project (ECG5000 dataset)

- **`references/study/`** — Lecture slides and supplementary readings: (Not provided in github)
  - `module1_part1_classical_timeseries.pdf`, `module1_part2_deep_learning.pdf` — Module 1: classical and deep-learning time series
  - `lecture1_autoencoders.pdf`, `lecture2_lstm_ae.pdf`, `lecture3_gans_graphs.pdf` — Lecture slides on autoencoders, LSTM-AE, GANs, and graphs
  - `LSTM_lecture_notes.pdf` — LSTM lecture notes
  - `sec1_2_invariance_equivariance.pdf`, `sec1_3_manifold_learning.pdf`, `sec1_4_gauge_theories.pdf` — Geometric deep learning sections
  - `graph_crash_course.pdf` — Graph neural networks crash course
  - `anogan.pdf` — AnoGAN reference paper
  - `solutions_module1.pdf` — Module 1 solutions

- **`references/atlas/`** — Project-specific instruction and schema files:
  - `course_guideline.md` — This document; course structure, objectives, and grading criteria
  - `elliptic_bitcoin_schema.md` — Elliptic Bitcoin dataset schema and feature documentation
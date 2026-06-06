To master Phase 1, we must move beyond a simple checklist. According to the course guidelines, your work should present a **clear narrative** and adopt a specific perspective rather than appearing as a list of methods.

Phase 1 is your **"Mathematical Manifesto."** You are setting the stage to prove that standard, Euclidean machine learning is insufficient for the complexity of financial crime.

---

## The Phase 1 Narrative: "The Geometric Bottleneck"

Your narrative for the examiners (and Professor Vaccarino) should be:

> _"Licit financial behavior is constrained by economic laws, forming a stable, low-dimensional manifold. Illicit actors, however, must violate these geometric constraints to hide their tracks. Standard models fail because they ignore the **relational topology**—they see the points, but they are blind to the 'shape' of the crime."_

### 1. Exploratory Data Analysis (EDA): Mapping the Terrain

Your EDA should highlight the salient aspects that influence your modeling choices, such as class imbalance and temporal structure.

- **Temporal Snapshot Analysis:** You must describe the 49 discrete time steps. Visualize the transaction volume over time to show that the "data manifold" is not static—it evolves.
    
- **Class Imbalance:** Clearly document that illicit transactions are a tiny minority. This justifies why you will later choose an **Anomaly Detection** framework (like an Autoencoder) rather than a simple classifier.
    
- **The "Licit Continent" vs. "Illicit Islands":** Use the "beautiful" visualizations we discussed. Instead of raw features, use a **Laplacian Eigenmap** to show that licit transactions cluster in a way that suggests an underlying manifold, while illicit ones are scattered as outliers.
    

### 2. The Baseline: Establishing the "Euclidean Failure"

The guidelines explicitly require at least one baseline to justify additional complexity.

- **The Setup:** Train a **Random Forest** or a **Logistic Regression** on only the 166 nodal features, completely ignoring the graph structure.
    
- **The Temporal Split:** Ensure your train, validation, and test split replicates a realistic production setting by respecting the **temporal order**. For example, train on the first 34 snapshots and test on the final 15.
    
- **The Justification:** Use this baseline to show that while features are helpful, they are not enough. This creates the "need" for the GNN you will build in Phase 2.
    

### 3. Latent Space Visualization: The "Aha!" Moment

To satisfy your desire for a "beautiful" and readable map, perform **Latent Space Mapping**:

- **Step:** Take the output of a simple 2-layer GCN (before it's fully trained) / or our implemented SGC (if possible) and project it using UMAP.
    
- **Aesthetic Strategy:** Apply **Kernel Density Estimation (KDE)** to create a "topographic map" of the latent space.
    

---

## The "Vaccarino" Checkpoint: Language of Phase 1

When writing your report, use these specific theoretical justifications:

- **Metric Choice:** Don't just use Accuracy; justify your metric (e.g., Precision-Recall AUC) against plausible alternatives based on the **cost of false positives** in a financial setting.
    
- **Methodological Rigor:** Explain that your EDA isn't just a "list of charts" but a systematic investigation into the **spectral properties** of the Bitcoin graph.
    


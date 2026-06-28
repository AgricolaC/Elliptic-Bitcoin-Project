## Exploratory Data Analysis: Network Homophily & Temporal Interaction

Graph Neural Networks rely heavily on the assumption of **homophily**—the principle that connected nodes tend to share the same label. We performed this edge-interaction analysis to explicitly measure whether this assumption holds true in the Bitcoin network. If illicit actors primarily transact with licit services (e.g., cashing out through an exchange), GNN message-passing might blur illicit signals rather than amplify them.

Based on the edge connectivity data in `results/eda_homophily.csv`, we analyze how different classes of nodes interact with one another over the 49 time steps (`tau`).

### 1. Aggregate Interaction Statistics

Summing the edge counts across all 49 time steps provides a clear picture of the network's macro structure:

| Interaction Type | Total Edges | Average per Time Step |
| --- | --- | --- |
| **Licit $\leftrightarrow$ Licit** | 33,930 | 692.4 |
| **Illicit $\leftrightarrow$ Unknown** | 5,451 | 111.2 |
| **Illicit $\leftrightarrow$ Licit** | 1,696 | 34.6 |
| **Illicit $\leftrightarrow$ Illicit** | 998 | 20.4 |

> [!NOTE]
> The vast majority of the network's confirmed edges exist entirely within the Licit economy. However, analyzing the edges connected to Illicit nodes reveals fascinating money-laundering topologies.

### 2. The Illicit Interaction Breakdown

When an illicit node transacts, where does the money go (or come from)? Out of the **8,145** total edges involving at least one illicit node:

* **To Unknown Nodes (66.92%)**: The vast majority of illicit interactions are with unlabelled nodes. This is expected; criminals use intermediary wallets, peel chains, and mixers to obfuscate the flow of funds before cashing out.
* **To Licit Nodes (20.82%)**: A significant portion of illicit transactions flow directly to licit nodes. This represents the **integration phase** of money laundering, where dirty funds are deposited into legitimate exchanges or services to be cashed out for fiat.
* **To Illicit Nodes (12.25%)**: Illicit nodes rarely interact with *other known* illicit nodes. This demonstrates **low homophily**—criminals generally avoid direct transactions with other known criminal entities to reduce the risk of chain-analysis deanonymization.

### 3. Temporal Burstiness (The "Campaign" Effect)

While illicit-illicit interactions average only ~20 per time step, they are highly concentrated in specific bursts. 

**Top 5 Time Steps for Illicit-Illicit Interactions:**
1. **$\tau = 29$**: 224 edges *(~22.4% of ALL illicit-illicit edges in one step!)*
2. **$\tau = 32$**: 119 edges
3. **$\tau = 24$**: 96 edges
4. **$\tau = 31$**: 60 edges
5. **$\tau = 20$**: 51 edges

#### Analytical Insights:
* **Event-Driven Activity**: The massive spike at $\tau = 29$ and the cluster around $\tau = 31, 32$ suggest specific, coordinated illicit events. In real-world terms, these bursts usually correspond to massive ransomware payouts, darknet market exit scams, or coordinated hack liquidations.
* **Feature Engineering Opportunity**: The temporal burstiness means that time itself (`tau`) is a crucial contextual feature. If a node is active during a known "burst" step (like $\tau = 29$), and connects to another node active in that step, the probability of it being illicit increases.

> [!TIP]
> **Modeling Recommendation**: Because of the low homophily (illicit nodes don't connect to illicit nodes), standard Graph Convolutional Networks (GCNs) that assume homophily (i.e., neighbor smoothing) might underperform. Models that can capture structural roles and anti-homophily (like GraphSAGE with specific aggregators, or GATs) or models that explicitly utilize the "Unknown" nodes as intermediaries will perform much better.

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

*Note on Base Rates:* While the raw count of Licit-Licit edges (33,930) seems to imply strong homophily, this is largely a statistical inevitability given that Class 0 outnumbers Class 1 by roughly 9 to 1. To definitively prove structural homophily beyond the baseline population skew, these counts must be compared against a null model of random connectivity. Nevertheless, analyzing the edge proportions connected to Illicit nodes reveals important structural constraints.

### 2. The Illicit Interaction Breakdown

When an illicit node transacts, where does the money go (or come from)? Out of the **8,145** total edges involving at least one illicit node:

* **To Unknown Nodes (66.92%) - The Projection Trap**: The vast majority of illicit interactions are with unlabelled nodes. These may represent mixers, coinjoins, or peel chains, but they may also be perfectly legitimate users who have not been forensically tagged. While tempting to project adversarial intent (assuming these are mixers or peel chains), `Unknown` simply means a lack of forensic evidence. If many of these are benign, un-tagged retail users, the true illicit-to-licit rate is vastly higher than 20%. We must be careful not to hallucinate criminal networks out of unlabelled interactions.
* **To Licit Nodes (20.82%)**: A portion of illicit transactions flow directly to known licit nodes, representing the **integration phase** of money laundering (depositing into legitimate exchanges to cash out).
* **To Illicit Nodes (12.25%)**: Illicit nodes interact relatively less with *other known* illicit nodes compared to licit nodes. This demonstrates **low homophily**—criminals generally avoid direct transactions with other known criminal entities to reduce the risk of chain-analysis deanonymization.

### 3. Temporal Burstiness (The "Campaign" Effect)

While illicit-illicit interactions average only ~20 per time step, they are highly concentrated in specific bursts. 

**Top 5 Time Steps for Illicit-Illicit Interactions:**
1. **$\tau = 29$**: 224 edges *(~22.4% of ALL illicit-illicit edges in one step!)*
2. **$\tau = 32$**: 119 edges
3. **$\tau = 24$**: 96 edges
4. **$\tau = 31$**: 60 edges
5. **$\tau = 20$**: 51 edges

**Algorithmic Batching vs. Coordinated Events**: While tempting to label the massive spike at $\tau = 29$ as a dramatic "hack" or "ransomware payout," this likely misinterprets Bitcoin's UTXO mechanics. A single large entity (like a darknet market) performing routine wallet maintenance—consolidating thousands of tiny incoming payments into a cold wallet during low-fee periods—generates hundreds of illicit-illicit edges instantly. 


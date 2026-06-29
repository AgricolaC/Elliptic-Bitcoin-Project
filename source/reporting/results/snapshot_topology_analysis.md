## Snapshot Topology Analysis

The defining event in this dataset is the **$\tau=43$ shock**, which corresponds to the sudden seizure and shutdown of the AlphaBay darknet market by international law enforcement in July 2017. AlphaBay was the largest darknet market at the time, and its abrupt removal drastically altered the landscape of illicit Bitcoin activity.

We analyze the macroscopic graph properties of the Bitcoin transaction network at each timestep ($\tau$). By tracking node volume, edge volume, mean degree, and graph density across the Pre-Shock ($\tau < 43$), Shock ($\tau = 43$), and Recovery ($\tau > 43$) phases, we aim to understand the physical impact of this shutdown on the network.

### 1. Structural Compaction(The Illusion of Macroscopic Stability)

At first glance, the snapshot metrics suggest that the AlphaBay shutdown did not break the macroscopic structure of the network:

* **Mean Degree**: During the Shock ($\tau=43$), the mean degree was a perfectly normal $2.35$ (consistent with the $2.05$ to $2.87$ pre-shock band).
* **Graph Density**: Density stayed tightly bounded between $0.0003$ and $0.0019$, showing no noticeable impact.

**The Reality Check**: Interpreting this as "stability" is a scaling artifact. Between $\tau=42$ and $\tau=43$, total nodes dropped from 7,140 to 5,063—a sudden **29% contraction of the entire network vertex set**. 

In graph theory, if a network loses nearly a third of its nodes in a single timestep yet maintains an identical mean degree, it requires an equally strong structural re-wiring. The remaining nodes must have quickly formed tight, localized sub-components to preserve the average degree distribution. The network did not "continue processing transactions with the exact same structural volume"; it underwent a **structural compaction**.

### 2. The Illicit Volume Crash (Prior Shift Confirmation)

While the *global* network structure remained stable, the *local* volume of illicit nodes collapsed entirely:

| Phase | $\tau$ | Total Nodes | Illicit Nodes | Illicit Rate |
|---|---|---|---|---|
| **Pre-Shock (Peak)** | $13$ | $4,528$ | $291$ | $35.9\%$ |
| **Pre-Shock (Late)** | $42$ | $7,140$ | $239$ | $11.0\%$ |
| **The Shock** | $43$ | $5,063$ | **$24$** | **$1.7\%$** |
| **Recovery (Trough)**| $46$ | $3,519$ | **$2$** | **$0.2\%$** |
| **Recovery (Late)** | $49$ | $2,454$ | $56$ | $11.7\%$ |

At $\tau=43$, illicit node volume dropped by $90\%$ overnight (from 239 to 24), and continued dropping to a mere 2 nodes by $\tau=46$. 
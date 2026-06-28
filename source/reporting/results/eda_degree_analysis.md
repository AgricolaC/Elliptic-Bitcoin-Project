## Exploratory Data Analysis: Node Degree Statistics

Before applying complex Graph Neural Networks or deep propagation mechanisms, it is essential to establish whether basic local graph topology can separate the classes. We performed this node degree analysis to answer a fundamental question: **Do illicit actors route funds differently than regular licit users?**

By analyzing the simple in-degree (incoming funds) and out-degree (outgoing funds) of each transaction node, we aim to uncover structural signatures of money laundering (such as peel chains or smurfing) versus typical licit behavior (such as exchange wallets or mining pool distributions). 

The following sections highlight the key structural differences between `class 0` (Licit) and `class 1` (Illicit) transactions in the Elliptic Bitcoin dataset.

> [!NOTE]
> The dataset exhibits a significant class imbalance. There are **42,019** nodes belonging to Class 0 compared to only **4,545** nodes in Class 1 (an approximate 9:1 ratio).

### 1. Out-Degree: The Key Differentiator

The most striking differences between the two classes lie in their out-degree distributions (the number of subsequent transactions a node sends funds to).

| Metric | Class 0 (Licit) | Class 1 (Illicit) |
| --- | --- | --- |
| **Mean** | 1.18 | 0.74 |
| **Std Dev** | 3.24 | 0.57 |
| **Max** | **472.0** | **3.0** |
| **Skewness** | 92.59 | 0.07 |
| **Kurtosis** | 12059.60 | -0.42 |

#### Analytical Insights:
* **Constrained Outflow for Illicit Nodes**: Illicit nodes have a strict upper bound on their out-degree (`max = 3`). This suggests that illicit transaction pathways do not fan out broadly. This behavior is highly characteristic of money laundering typologies like **peel chains**, where funds are linearly moved with one output going to a target and another going to a change address. 
* **Presence of "Hubs" in Licit Nodes**: Class 0 nodes exhibit massive right-tail outliers (`max = 472`, `kurtosis = ~12060`). This indicates the presence of exchange wallets, mining pools, or services that distribute funds to many different addresses simultaneously.

### 2. In-Degree: Heavy-Tailed Similarities

The in-degree distributions (how many transactions feed into a node) share more similarities across classes but still contain subtle differences.

| Metric | Class 0 (Licit) | Class 1 (Illicit) |
| --- | --- | --- |
| **Mean** | 1.91 | 1.27 |
| **Std Dev** | 7.12 | 7.21 |
| **Median (50%)** | 1.0 | 1.0 |
| **Max** | 284.0 | 177.0 |

#### Analytical Insights:
* **Scale-Free Network Properties**: Both classes exhibit right-skewed, heavy-tailed distributions (skewness > 14, kurtosis > 300). Most nodes receive exactly 1 transaction (the median and 75th percentile are both `1.0` for both classes).
* **Consolidation**: Both classes have nodes that consolidate funds from many sources (max in-degree 284 for Class 0, and 177 for Class 1). For illicit actors, this could represent the consolidation phase of money laundering where funds scattered across many addresses are swept into a single deposit address.

### 3. In-Degree / Out-Degree Correlation

* **Class 0**: `-0.015`
* **Class 1**: `-0.105`

Both classes show a slightly negative correlation between in-degree and out-degree. For illicit nodes, this negative correlation is stronger. When illicit nodes consolidate funds from many inputs (high in-degree), they almost never fan them out to multiple outputs (low out-degree).

### Conclusion & Next Steps

The topology of the transaction graph provides a highly discriminatory signal:
1. **Illicit transactions are structurally constrained** downstream (out-degree $\le$ 3).
2. **Licit transactions naturally form hubs** (out-degree up to 472).

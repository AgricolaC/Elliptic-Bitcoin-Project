# Exploratory Data Analysis: PageRank Statistics

Based on the statistical summary in `results/eda_pagerank_stats.csv`, we can analyze the centrality and influence of nodes belonging to `class 0` (Licit) and `class 1` (Illicit) within the transaction network using their PageRank scores.

## 1. Distribution Overview

| Metric | Class 0 (Licit) | Class 1 (Illicit) |
| --- | --- | --- |
| **Mean** | $2.72 \times 10^{-4}$ | $2.22 \times 10^{-4}$ |
| **Std Dev** | $7.77 \times 10^{-4}$ | $7.46 \times 10^{-4}$ |
| **Median (50%)** | $1.18 \times 10^{-4}$ | $1.33 \times 10^{-4}$ |
| **Max** | $2.65 \times 10^{-2}$ | $2.50 \times 10^{-2}$ |
| **Skewness** | 12.79 | 19.63 |
| **Kurtosis** | 243.51 | 468.19 |

### Analytical Insights:
* **Higher Median for Illicit Nodes**: Surprisingly, the median PageRank for Illicit nodes ($1.33 \times 10^{-4}$) is higher than for Licit nodes ($1.18 \times 10^{-4}$). The 25th percentile is also higher for Illicit nodes. This implies that the "typical" illicit node is slightly more central or receives more directed flow than the "typical" licit node. This could be due to the chain-like structure of money laundering (e.g., peel chains) where centrality is preserved along the path.
* **Higher Mean for Licit Nodes**: While the median is lower, the mean PageRank for Licit nodes is higher. This indicates that the right tail of the Licit distribution contains nodes with exceptionally high PageRank scores that pull the mean upward.

## 2. The Right Tail: Hubs of Influence

Let's look at the upper percentiles to understand the most influential nodes in each class.

| Percentile | Class 0 (Licit) | Class 1 (Illicit) |
| --- | --- | --- |
| **75%** | $2.17 \times 10^{-4}$ | $2.04 \times 10^{-4}$ |
| **95%** | $7.47 \times 10^{-4}$ | $4.70 \times 10^{-4}$ |
| **99%** | $3.14 \times 10^{-3}$ | $8.36 \times 10^{-4}$ |

### Analytical Insights:
* **Licit "Whales" Dominate the Top**: At the 95th and 99th percentiles, Licit nodes have significantly higher PageRank scores than Illicit nodes. The 99th percentile for Licit nodes ($3.14 \times 10^{-3}$) is nearly 4x higher than for Illicit nodes ($8.36 \times 10^{-4}$). 
* **Alignment with Degree Findings**: This perfectly aligns with our previous finding that Licit transactions naturally form massive hubs (like exchanges). These high-degree hubs naturally accumulate the highest PageRank in the network.
* **Extreme Outliers in Illicit Nodes**: Despite the 99th percentile being relatively low, the maximum PageRank for Illicit nodes ($2.50 \times 10^{-2}$) is very close to the maximum for Licit nodes ($2.65 \times 10^{-2}$). This single extreme outlier causes the massive kurtosis (468.19) in the Illicit distribution. This might represent a major darknet marketplace or a significant point of consolidation before cashing out.

## Conclusion

PageRank reveals a nuanced structural difference between the two classes:
1. **The "Average" Illicit Node**: Tends to have slightly higher centrality than an average licit node, likely due to funds being funneled through structured linear chains.
2. **The "Elite" Licit Node**: The top 1-5% of the most influential nodes are overwhelmingly Licit. These are the major structural pillars of the network.


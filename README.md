# Credit Risk Model

## Credit Scoring Business Understanding

1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?  
   The Basel II Accord mandates robust risk measurement and transparency for regulatory compliance. Interpretable models, such as Logistic Regression with Weight of Evidence (WoE), enable clear explanation of risk predictions, aligning with regulatory standards, facilitating audits, and fostering stakeholder trust. Comprehensive documentation ensures traceability, supporting Basel II’s risk management requirements.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?  
   Without a "default" label in the dataset, a proxy variable (e.g., derived from RFM clustering) is essential to estimate credit risk for model training. An inaccurate proxy may misclassify customers, leading to erroneous loan approvals or rejections. Business risks include financial losses from approving high-risk loans, missing opportunities with low-risk customers, or regulatory penalties due to unreliable predictions.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?  
   - Logistic Regression with WoE: Highly interpretable, meets regulatory demands, and is easier to explain to stakeholders, but may underfit complex data, reducing predictive accuracy.  
   - Gradient Boosting: Offers superior accuracy and captures complex patterns, but its lower interpretability complicates regulatory compliance and stakeholder communication. In regulated financial contexts, interpretability often outweighs marginal performance gains.
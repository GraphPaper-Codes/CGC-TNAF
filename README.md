# CGC-TNAF

# Contrastive Graph Clustering with a Topology-sensitive Noise Augmentation Framework

Mohammad Saeb Nahi, M. A. Balafar, Jafar Tanha and Nazila Pourhaji Aghayengejeh

we propose a novel Contrastive Graph Clustering with a Topology-sensitive Noise Augmentation Framework (CGC-TNAF) that consists of two main modules: Topology-aware Hybrid Noise Augmentation (THNA) and Contrastive Learning with Topology-aware Noise (CLTN). The THNA module combines graph components corresponding to high and low frequencies to model global similarities and finegrained node distinctions, while introducing Gaussian noise that considers the topology as useful perturbation to enhance the representation diversity. The CLTN module introduces a novel mechanism for generating exclusive, negative samples guided by noise, enabling the construction of triplet contrastive pairs (Target, Positive, Negative) based on topology.

![](https://github.com/GraphPaper-Codes/CGC-TNAF/image/Capture1.JPG)
<div align=center>
Figure 1: The overall framework of CGC-TNAF.
</div>

### Start

- Step1: unzip the dataset into the **./dataset** folder
- Step2: run train.py

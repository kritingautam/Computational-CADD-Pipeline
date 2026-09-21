\# Graph Neural Network (GNN) for Drug-Target Interaction (DTI) Classification

&nbsp;

An end-to-end computational biology and deep learning pipeline designed to screen chemical compounds for drug-target interaction metrics using Graph Neural Networks (GNNs). This project bridges clinical pathology literacy with predictive bioinformatics, leveraging \*\*PyTorch Geometric\*\* and \*\*Computer-Aided Drug Design (CADD)\*\* principles to identify molecular binders and structural toxicological indicators.

&nbsp;

\#\# 🧬 Scientific & Clinical Rationale

Traditional high-throughput screening of molecular libraries is resource-intensive. By treating chemical compounds as molecular graphs—where atoms are nodes and covalent chemical bonds are edges—we can pass spatial and chemical messages across a molecule's topological layout.&nbsp;

&nbsp;

This model accelerates early-stage translation pipelines by filtering out reactive or incompatible ligands \*in silico\* before moving physical assays into wet lab environments. High-scoring candidates from this deep learning classifier are exported for structured 3D docking validation using \*\*Schrödinger Maestro\*\* workflows.

&nbsp;

\#\# 📊 Atom Featurization Parameters

In the \*\*Tox21 dataset\*\* parsed via PyTorch Geometric’s standard MoleculeNet loader, each individual atom (node) is characterized by a vector of \*\*9 foundational biochemical properties\*\*. Instead of just storing the raw element name, the framework leverages \*\*RDKit featurizers\*\* to translate raw chemical structures into multi-dimensional numerical tensor arrays.&nbsp;

&nbsp;

The 9 properties embedded within the node feature matrix ($x$) are detailed below:

\* \*\*Atom Type / Element:\*\* Mapped via a multi-element one-hot encoding configuration.

\* \*\*Chirality:\*\* The 3D spatial orientation of the atom (Right-handed \*R\*, Left-handed \*S\*, or non-chiral).

\* \*\*Formal Charge:\*\* The physical electrical charge of the atom for calculating electrostatic properties.

\* \*\*Partial Charge:\*\* Subtle local electron distribution calculated across a molecule's surface.

\* \*\*Aromaticity:\*\* Binary indicator (0 or 1\) identifying if the atom is part of a resonance-stabilized ring system.

\* \*\*Hybridization State:\*\* The chemical orbital bonding shape ($sp$, $s{p}^{2}$, $s{p}^{3}$, etc.) defining spatial geometry.

\* \*\*Hydrogen Bonding:\*\* Vector tracking whether the specific atom functions as a Donor or Acceptor.

\* \*\*Radical Electrons:\*\* The count of unpaired valence electrons on the atom signaling reactivity.

\* \*\*Degree / Valence:\*\* The total number of immediate neighbor atoms this node forms covalent bonds with.

&nbsp;

\#\# 💻 Installation & Environment Setup

To run this pipeline locally, set up your Python execution sandbox environment and install the verified dependencies:

&nbsp;

\`\`\`bash

\# Clone the repository

git clone https://github.com

cd GNN-Drug-Target-Interaction

&nbsp;

\# Install dependencies

pip install \-r requirements.txt

&nbsp;

\# Run the execution pipeline

python pipeline.py

\`\`\`

&nbsp;

&nbsp;
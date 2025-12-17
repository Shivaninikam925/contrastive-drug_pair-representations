# Contrastive Representation Learning for Drug Pairs

This project explores self-supervised contrastive learning to obtain
interaction-aware representations of drug pairs without relying on
explicit drug–drug interaction (DDI) labels.

## Motivation
Most DDI models are trained using supervised labels that are sparse,
noisy, and task-specific. Instead of predicting interactions directly,
this project focuses on learning robust representations of drug pairs
using contrastive self-supervision, enabling better transfer to
downstream biomedical tasks.

## Core Idea
Each drug pair is represented through multiple augmented views
(e.g., feature masking, noise injection). A contrastive objective
encourages representations of the same pair to be close while pushing
apart representations of different pairs. This induces semantic
structure without explicit interaction supervision.

## Method Overview
- Drug-level encoder: small MLP
- Pair representation: concatenation of encoded drug embeddings
- Training objective: InfoNCE (NT-Xent) contrastive loss
- No interaction labels used during training

## Results
UMAP visualizations show that contrastive training produces structured
embedding geometry compared to random embeddings, despite the absence
of supervision.

## Significance
This project demonstrates how self-supervised representation learning
can be applied to drug interaction modeling, offering a scalable
pretraining approach for biomedical data where labels are unreliable.

## Extensions
This framework can be extended to:
- Real molecular fingerprints or graph-based encoders
- Downstream DDI prediction with frozen embeddings
- Biological validation using known interaction categories

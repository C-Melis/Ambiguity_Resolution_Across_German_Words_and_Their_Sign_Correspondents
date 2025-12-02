# Ambiguity Resolution Across German Words and Their Sign Correspondents
## Melis Çelikkol, Master's Project

This repository contains the script and data for analyzing lexical ambiguity correspondence patterns between German (spoken language) and Deutsche Gebärdensprache/DGS (German Sign Language).

## Overview

We investigate how lexical ambiguities in spoken languages map to those in sign languages by matching German words from Word Usage Graph (WUG) datasets to their DGS sign correspondents in the Digital Dictionary of German Sign Language (DW-DGS).

## Key Components

- **Manual Annotations**: Word-to-sign ID mappings capturing cross-modal semantic correspondence patterns
- **Computational Methods**: Transformer-based semantic similarity systems for predicting correspondence patterns
- **Pathway Analysis**: Classification of three main correspondence types (one-to-many, many-to-one, one-to-one)

## Data

The data files include manually annotated mappings showing diverse correspondence patterns:
- German words corresponding to multiple distinct DGS signs (e.g., *erlauben* → 3 signs)
- Single DGS signs corresponding to multiple German words (e.g., one sign → *Abend* and *Nacht*)
- Parallel ambiguity structures (e.g., *Westen* and its sign correspondent)

## Data Files

Our annotation process produces complementary data files serving distinct purposes:

**Manual Annotation Files**: Contains word-sentence-video_id mappings representing contextualized word uses paired with their corresponding DGS sign identifiers. These files enable the prediction system to learn from annotated word-to-sign correspondences.

**Sense and Sign ID Files**: Contains meaning-video_id mappings, where the meaning column combines German translations with sense clarifications for each sign. These files provide the semantic descriptions that enable similarity-based matching. An additional variant containing only German translations (without sense clarifications) is used for ablation testing.

**Pathway Ground Truth Files**: Contains word-pathway-video_id mappings representing our classification of each German word into pathway types based on manual semantic analysis. These files serve exclusively for evaluating whether computational models can independently discover the same correspondence patterns that human annotators identified.

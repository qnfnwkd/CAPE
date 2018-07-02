# Content-Aware Hierarchical Point-of-Interest Embedding Model for Successive POI Recommendation
Recommending a point-of-interest (POI) a user will visit next based on temporal and spatial context information is an important task in mobile-based applications. Recently, several POI recommendation models based on conventional sequential-data modeling approaches have been proposed. However, such models focus on only a user’s checkin sequence information and the physical distance between POIs. Furthermore, they do not utilize the characteristics of POIs or the relationships between POIs. To address this problem, we propose CAPE, the first content-aware POI embedding model which utilizes text content that provides information about the characteristics of a POI. CAPE consists of a check-in context layer and a text content layer. The check-in context layer captures the geographical influence of POIs from the check-in sequence of a user, while the text content layer captures the characteristics of POIs from the text content. To validate the efficacy of CAPE, we constructed a large-scale POI dataset. In the experimental evaluation, we show that the performance of the existing POI recommendation models can be significantly improved by simply applying CAPE to the models.

## Model description
<p align="left">
<img src="/figures/context_layer.png" width="400px" height="auto">
</p>
Context Layer
<p align="left">
<img src="/figures/content_layer.png" width="400px" height="auto">
</p>
Content Layer

## Dataset
Dataset is available at [here](https://dmis.korea.ac.kr/cape).

## Getting Started
To run the implements, it is required to preprocess the above dataset.
For the context layer, each line consists of POIs visited by each user.
For the content layer, each line consists of a POI and words which are most frequently used in the POI.
The preprocessed data examples are included in the toyset folder.

### Prerequisites
- python 2.7
- PyTorch 0.4.0

### Usage
```bash
git clone https://github.com/qnfnwkd/CAPE
cd CAPE
python train.py
```

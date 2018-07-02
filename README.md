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

## Data set
Data set is available at [here](https://s3.amazonaws.com/poiprediction/instagram.tar.gz). The data set includes "train.txt", "validation.txt", "test.txt", and "visual_feature.npz". The "train.txt"  "validation.txt" "test.txt" files include the training, validation, and tesing data respectively. The data is represented in the following format:
```bash
<post_id>\t<user_id>\t<word_1 word_2 ... >\t<poi_id>\t<month>\t<weekday>\t<hour>
```

All post_id, user_id, word_id, and poi_id are anonymized. Photo information also cannot be distributed due to personal privacy problems. So we relase the converted visual features from the output of the FC-7 layer of [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) used as the visual feature extractor. If you want to use other visual feature extractor, such as [GoogleNet](http://arxiv.org/abs/1602.07261), [ResNet](https://arxiv.org/abs/1512.03385), you could implement it on your source code. We use a pre-trained VGGNet16 by [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) The "visual_feature.npz" file contains the visual features where the i-th row denotes i-th post's features.

### statistics
<table style="align=center;">
<tr><td>number of total post</td><td>number of POIs</td><td>number of users</td><td>size of vocabulary</td></tr>
<tr><td>736,445</td><td>9,745</td><td>14,830</td><td>470,374</td></tr>
<tr><td>size of training set</td><td>size of validation set</td><td>size of test set</td></tr>
<tr><td>526,783</td><td>67,834</td><td>141,828</td></tr>
</table>

## Getting Started
The code that implements our proposed model is implemented for the above dataset, which includes pre-processd visual feature. If you want to use a raw image that is not pre-processed, implement VGGNet on your source code as visual CNN layer.

### Prerequisites
- python 2.7
- tensorflow r1.2.1

### Usage
```bash
git clone https://github.com/qnfnwkd/DeepPIM
cd DeepPIM
python train.py
```

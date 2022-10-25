# AI Workloads resource utilization characterization 

## Goal
Predicting task Interference for GPU Sharing

Reduce Individual Job Slow Down while Maximizing Packing Saving when two job runs on a single GPU, concurrently. 

Target Variables: Maximum GPU Utilization Percent, and Maximum GPU Memory Allocated Percent 

## AI workloads description 
### Case 1: 
Image Classification on German Traffic Sign Recognition Benchmark (GTSRB) using standard ML architectures: MobileNet, NasnetMobile, EfficientNetV2, ResNet50, and InceptionV3. 

Code file: [Image_Classification.py](./Image_Classification.py)

### Case 2: 
Image Segmentation on Oxford-IIIT Pet Dataset (v-3.2.0) 

Code File: [Image_Segmentation.py](./Image_Segmentation.py)


### Case 3: 
Reinforcement Learning - Cart-Pole problem in OpenAI Gym suite environment 

Code File: [RL_CartPole.py](./RL_CartPole.py)


### Case 4: 
Generative Adversarial Networks (GAN) - Deep Convolutional GAN for generating synthetic MNIST digits. 

Code File: [DCGAN_MNIST.py](./DCGAN_MNIST.py)

### Case 5: 
Natural Language Processing - Various BERT models fine-tuned for sentiment analysis on IMDB movie review dataset. 

Code File: [NLP_BERT.py](./NLP_BERT.py)

## Experiments Env
Executed in Google Colaboratory Pro platform, Tesla P100 GPU. All implementation in TensorFlow. 

## Collected metrics and format
Pls refer to attached files

![Dataset Description](./images/Dataset_Desc.png)

Collected Dataset: [Dataset.csv](./Dataset.csv)

## Feature Selection
Through coefficient of determination (Pearson, Spearman, Kendall), decide the most relevant features. 


## Synthetic Data Generation 

Generation Strategy: Synthetic data is generated for selected attributes wrt Batch Size. Different interpolation techniques like Linear, Pchip, Barycentric, Krogh, Cubicspline, and Spline of order=2 is explored. For quality assessment of synthetic samples, we consider synthetic samples as pseudo-ground truth and re-generated collected data. Root Mean Square Error (RMSE), Mean Square Error (MSE), and Mean Absolute Error (MAE) metrics are used for quality assessment. 


![Synthetic Data Generation Design Methodology](./images/Synthetic_Data_Gen_Design.png)

## Clustering 
K-means clustering (with elbow method and distortion score) with relevant features and synthetic data. 
![Clustering Results](./images/Clustering.png)


## Evaluating Task Interference
Evaluation Metrics: Individual Slow Down, Packing Saving. 
Scheduler can be designed to prioritize either of the metrics, based on user needs.



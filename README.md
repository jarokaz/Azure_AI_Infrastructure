# Azure AI Infrastructure - running AI workloads at scale workshop

Azure AI infrastructure provides virtually limitless scale to train and operationalize the largest machine learning models. In this workshop, we focus on training and operationalizing large scale deep learning models. The goal  services including Azure GPU VMs, Azure Batch AI, and Azure Container Services. Through hands-on labs and instructor led walkthroughs the attendees will master the following scenarios:
-  Single GPU Node  Model Training
-  Parallel training using multi GPU VMs 
-  Distributed training on clusters of GPU VMs
-  Cloud model deployment & management using Azure Machine Learning Model Management
-  Edge model deployment to IoT Edge and other edge platforms 

When run in a classroom setting a typical agenda goes as follows:
* Introductions
* Azure Cloud AI Platform Overview
* Hands-on Labs
  * Lab 1 - Warm-up - Single GPU training
  * Lab 2 - Parallel training on multi GPU VMs
  * Lab 3 - Distributed training using **asynchronous SGD** and parameter server architecture
  * Lab 4 - Distributed training using **synchronous Allreduce** approach
  
  
## Repo folder structure
- DLCheetsheets - The Jupyter notebook summarizing key Deep Learning concepts. 
- Presentations - PDF copies of the presentations used by an instructor to do an overview of AI Infrastructure services
- Labs - Hands on labs - the gist of the workshop
- Utils - Python scripts to pre-process the CIFAR10 and TINY-IMAGENET data sets used during the workshop. The pre-processed data sets have been upload to Azure storage public containers so the scripts are for reference.




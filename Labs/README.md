## Hands-on labs

The hands-on labs are the gist of the workshop. The goal of the labs is to develop a solid understanding and practical experience on how to utilize Azure Batch AI, Azure GPU VMs, and Azure Machine Learning Model Management to train and operationalize large scale deep learning models on Azure. All labs utilize the same dataset and model: Tiny Imagenent and ResNet 50. Each lab focuses on a specific training or deployment approach.
- Lab 1 - the goal of this lab is to develop a basic understanding of Azure Batch AI and Azure GPU VMs. The training scenario is very simple - single GPU on a single node. 
- Lab 2 - the second lab extends the model and training regime from Lab 1 to distributed training using the parameter server architecture
- Lab 3 - the third lab focuses on parallel training - multiple GPUs on a single node
- Lab 4 - the fourth lab brings together skills developed in the previous labs into the most complex and powerfull scenario - multiple nodes with multiple GPUs.
- Lab 5 - demonstrates how to deploy a deep learning model using Azure Machine Learning Model Management service.

As noted, all labs use the same data set and model. The model is based on a ResNet 50 architecture as described in the seminal paper *Deep Residual Learning for Image Recognition* (Keiming He et al., 2015).  As described in the paper, the model was designed around 224x224x3 inputs.

The data set utilized in the workshop is Aerial Imagery.

The choice of ResNet 50 and Tiny Imagenet attempts to strike a balance between non-trivial and feasible scenarios. We wanted to go beyond a toy scenario (E.g. a simple convolutional net and MNIST) to demonstrate the value of powerful compute services in Azure. At the same time we wanted to limit computational complexity due time and budget constraints of the workshop.

All labs utilize TensorFlow and Keras. Lab 2 and 3 utilize Uber's Horovod API. In future we may extend the labs to other deep learning frameworks.

*You can use a variety of tools and APIs to manage Azure resources, including Azure Portal and PowerShell. In this workshop, we decided to use Azure CLI.* 

*When going through the labs you will be manually executing sequences of Azure CLI commands. It is important to note that in production environments **you should script and parameterize** repeatable workflows using a scripting environment of your choice. E.g. bash, python, etc.*


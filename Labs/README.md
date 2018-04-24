## Hands-on labs

The hands-on labs are the gist of the workshop. The goal of the labs is to develop a solid understanding and practical experience on how to train and operationalize large scale deep learning models on Azure. All labs utilize the same dataset and model: Tiny Imagenent and ResNet 50. Each lab focuses on a specific training or deployment approach.
- Lab 1 - the goal of this lab is to develop a basic understanding of Azure Batch AI and Azure GPU VMs. The training scenario is very simple - single GPU on a single node. 
- Lab 2 - the second lab extends the model and training regime from Lab 1 to distributed training using the parameter server architecture
- Lab 3 - the third lab focuses on parallel training - multiple GPUs on a single node
- Lab 4 - the fourth lab brings together skills developed in the previous labs into the most complex and powerfull scenario - multiple nodes with multiple GPUs.
- Lab 5 - demonstrates how to deploy a deep learning model using Azure Machine Learning Model Management service.

As noted, all labs use the same data set and model. The model is based on a ResNet 50 architecture as described in the seminal paper *Deep Residual Learning for Image Recognition* (Keiming He et al., 2015). The model used in the workshop has been modified to work with 64x64x3 images. The model described in the paper was designed around 224x224x3 inputs.

The data set utilized in the workshop is Tiny Imagenet. Tiny ImageNet Challenge is the default course project for Stanford CS231N course in deep learning. It runs similar to the ImageNet challenge (ILSVRC). Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.

The choice of ResNet 50 and Tiny Imagenet attempts to strike a balance between non-trivial and feasible scenarios. We wanted to go beyond a toy scenario (E.g. a simple convolutional net and MNIST) to demonstrate the value of powerful compute services in Azure. At the same time we wanted to limit computational complexity due time and budget constraints of the workshop.

All labs utilize TensorFlow. ResNet 50 is defined in tf.keras. Training regimes are defined using tf.Estimator API. Lab 3 and 4 utilize Uber's Horovod API. In future we may extend the labs to other deep learning frameworks.


# Azure AI Infrastructure - running AI workloads at scale workshop

Azure AI infrastructure provides virtually limitless scale to train and operationalize the largest machine learning models. In this workshop, we focus on training and operationalizing large scale deep learning models. The key technologies covered during the workshop are Azure GPU VMs, Azure Batch AI, and Azure Machine Learning Model Management. Through hands-on labs and instructor led walkthroughs the attendees will master the following scenarios:

-  Parallel training using multi GPU VMs 
-  Distributed training on clusters of GPU VMs
-  Cloud deployment & management of machine learning models
 
 A typical agenda goes as follows:
* Introductions
* Azure Cloud AI Platform Overview
* Hands-on Labs
  * Lab 0 - Environment set up
  * Lab 1 - Single GPU training
  * Lab 2 - Parallel training on a multi GPU VM 
  * Lab 3 - Distributed training on a cluster of GPU VMs 
  * Lab 4 - Model operationalization on Azure Kubernetis Server
  * Lab 5 - Model operationalizetion on Azure FPGA service 
  
  
## Repo folder structure
- DLCheetsheets - The Jupyter notebook summarizing key Deep Learning concepts. 
- Presentations - Azure Cloud AI Platform Overview presentation PDF
- Labs - Hands on labs - the gist of the workshop


## Workshop environment set up
To participate in the workshop you need a workstation with  `ssh`, `git`, text editor and the latest version of `Azure CLI 2.0`. 

### Install pre-requisites on Windows 10

Although you can use any `ssh` and `git` clients, and you can install `Azure CLI` on Windows command prompt, we highly recommend using  Ubuntu on Windows Subsystem for Linux.

Install Windows Subsystem for Linux and Ubuntu distributions

https://docs.microsoft.com/en-us/windows/wsl/install-win10

Install `Azure CLI` for Ubuntu.

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-apt?view=azure-cli-latest

If you choose to use Window command prompt follow this instructions to install `Azure CLI`

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?view=azure-cli-latest

### Install pre-requisites on Mac OS

Install Homebrew

https://docs.brew.sh/Installation.html

Install git
```
brew install git
```

Install `Azure CLI 2.0`

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-macos?view=azure-cli-latest

### Azure Cloud Shell
Azure Cloud Shell includes all pre-requisites required for the workshop.

### Azure Data Science VM
Azure Data Science Virtual Machine includes all pre-requisites. 


## Create the workshop's resource group and storage
All Azure resources created during the workshop will be hosted in the same resource group. It will simplify navigation and clean-up. This streamlined approach will work well for the workshop but does not represent the best practice for more complex production deployments. Refer to you organization's Azure guidance when setting up production grade environments.


### Login to your Azure subscription
```
az login
```
If you have multiple subscriptions set the right one with
```
az account set -s <subscription ID>
```
### Register Batch AI resource providers
Make sure that Batch AI resource providers are registered for you subscription. This is a one-time configuration.
```
az provider register -n Microsoft.BatchAI
az provider register -n Microsoft.Batch
```
### Create a resource group

```
az group create --name <resource group name> --location <location>
az configure --defaults group=<resource group Name>
az configure --defaults location=<location>
```

### Create a storage account and azure file share
We will use an Azure file share backed up by  Azure storage to store training data, training scripts, training logs, checkpoints, and the final model.
```
az storage account create --name <storage Account Name> --sku Standard_LRS
```
All labs in the workshop utilize Azure File Shares as shared storage. As noted by the instructor other shared storage options (e.g. NFS and distributed file systems) may perform better for really large data sets.

```
az storage share create \
    --account-name <Storage account Name> 
    --name <File share name>
```

#### Create data and scripts directories in the share
```
az storage directory create \
    --share-name  <File share name>
    --name data
    
az storage directory create \
    --share-name  <File share name>
    --name scripts
```

#### Copy training data
The training data in the TFRecords format have been uploaded to a public container in Azure storage. Use the following command to copy the files to your file share. The `--dryrun` option allows you to verify the configuration before starting the asynchronous copy operation.

The instructor will provide you with <Storage account access key>
    

```
az storage file copy start-batch \
  --destination-path data \
  --destination-share <File share name> \
  --source-account-name azaiworkshopst \
  --source-account-key <Storage account access key> \
  --source-container tinyimagenet \
  --pattern '*' \
  --dryrun
```

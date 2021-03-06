# Environment set up

## Configure Azure Data Science Virtual Machine
To participate in the workshop you need a workstation with  `ssh`, `git`, text editor, the latest version of `Azure CLI 2.0`, and `azcopy`. 

You can install these components on Windows, MacOS, or Linux. 

For the purpose of the workshop we will use Azure Data Science VM as an attendee workstation. The instructor will guide you through the provisioning process.

Azure Data Science Virtual Machine comes with the required pre-requistes installed. However; to make sure that you have the latest version of `azure-cli` and `azcopy` we recommend to do fresh installs.

To install `azure-cli`
```
conda create -n <environment name> python=3.6 
source activate <environment name>
pip install azure-cli
```
To install `azcopy`

```
wget -O azcopy.tar.gz https://aka.ms/downloadazcopylinux64
tar -xf azcopy.tar.gz
sudo ./install.sh
```

If you want to monitor the jobs using Tensorboard you need to open port 6006 on your virtual machine.

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

### Create a storage account 
We will use an Azure file share backed up by  Azure storage to store training data, training scripts, training logs, checkpoints, and the final model.
```
az storage account create --name <storage account name> --sku Standard_LRS
```
All labs in the workshop utilize Azure File Shares as shared storage. Note that other shared storage options (e.g. NFS and distributed file systems) may perform better for really large data sets.

To avoid entering the storage account name and the storage account key on each command we can store them in an environmental variables

```
az storage account keys list \
    -n <Storage account name> \
    -o table
export AZURE_STORAGE_ACCOUNT=<storage account name>
export AZURE_STORAGE_ACCESS_KEY=<storage account access key>
export AZURE_BATCHAI_STORAGE_ACCOUNT=<storage account name>
export AZURE_BATCHAI_STORAGE_KEY=<storage account access key>

```


### Create an azure file share
```
az storage share create \
    --account-name <storage account Name> 
    --name <File share name>
```

#### Create data and scripts directories in the share
```
az storage directory create \
    --share-name  <file share name>
    --name data
    
az storage directory create \
    --share-name  <file share name>
    --name scripts
```

#### Copy training data
The data used during the workshop have been uploaded to a public Windows Storage container. You will need to copy it to the `data` folder you created in the previous step



```
azcopy \
--source https://azureailabs.blob.core.windows.net/woodtfrecords \
--destination https://<storage account name>.file.core.windows.net/<file share name>/data \
--dest-key <storage access key> \
--recursive
```

## Prepare GPU cluster

Note that you have a lot flexibility configuring clusters. For the purpose of the workshop, we are configuring a very simple cluster with two multi-gpu nodes, a single shared storage end-point, and ssh key based authentication

```
az batchai cluster create \
  --name  <Cluster name> \
  --location eastus2
  --vm-size STANDARD_NC6 \
  --image UbuntuLTS \
  --min 1 \
  --max 1 \
  --afs-name <File share name> \
  --afs-mount-path external \
  --generate-ssh-keys \
  --user-name $USER 
```

### Get cluster status
```
az batchai cluster list -o table
```

### List ssh connection info for the nodes in a cluster
```
az batchai cluster list-nodes -n <Cluster name> -o table
```

### Explore the cluster's node
```
ssh <IP address> -p node
cd /mnt/batch/tasks/shared/LS_root/mounts
```


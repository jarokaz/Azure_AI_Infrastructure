# Using Batch AI to run a training job on a single GPU

## Register Batch AI resource providers
Make sure that Batch AI resource providers are registered for you subscription. This is a one-time configuration.
```
az provider register -n Microsoft.BatchAI
az provider register -n Microsoft.Batch
```
## Create a resource group
Batch AI clusters and jobs are Azure resources and must be placed in an Azure resource group
```
az group create --name <Resource Group Name> --location westus2
az configure --defaults group=<Resource Group Name>
az configure --defaults location=westus2
```

## Create a storage account
We will use an Azure file share backed up by  Azure storage to store training data, training scripts, training logs and checkpoints, and the final model.
```
az storage account create --name <Storage Account Name> --sku Standard_LRS
```

## Prepare Azure file share
### Create a file share
```
az storage share create \
    --account-name <Storage Account Name> 
    --name baifs
```
### Retrieve storage account keys
```
az storage account keys list \
    -n <account name> \
    -g <resource group name> \
    -o table
```

### Set environmnent variables
```
export AZURE_STORAGE_ACCOUNT=<account name>
export AZURE_STORAGE_ACCESS_KEY=<access key>
```

### Create a directory in the share
```
az storage directory create \
    --share-name  baifs
    --name lab01
```

### Upload training data
```
cd ~/repos/AzureAIInfrastructure/data/cifar10
az storage file upload --share-name baifs --source train.tfrecords --path lab01
az storage file upload --share-name baifs --source validation.tfrecords --path lab01
cd ~/repos/AzureAIInfrastructure/Labs/Lab01-WarmUp
az storage file upload --share-name baifs --source model.py --path lab01
az storage file upload --share-name baifs --source train.py --path lab01
```

### Verify that files are in the folder
```
az storage file list --share-name baifs --path lab01 -o table
```

## Prepare a GPU cluster
### Create a single GPU VM node
```
az batchai cluster create \
  --name  <cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuLTS \
  --min 1 \
  --max 1 \
  --storage-account-name <storage account name> \
  --afs-name baifs \
  --afs-mount-path azurefileshare \
  --user-name <user name> \
  --password <password>
```

### Get cluster status
```
az batchai cluster list -o table
```

## Create a training job

Create a JSON template file for job `job.json'`

### Create job
```
az batchai job create \
  --name mytfjob \
  --cluster-name <cluster name> \
  --config job.json
```
### Monitor job
```
az batchai job list -o table
```

### List stdout and stderr output
```
az batchai job list-files --name mytfjob --output-directory-id stdouterr
```

### Delete the job
```
az batchai job delete --name <job name>
```

### Delete the  cluster
```
az batchai cluster delete --name <cluster name>
```



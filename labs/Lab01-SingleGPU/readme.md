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
az group create --name <Resource group name> --location westus2
az configure --defaults group=<Resource group Name>
az configure --defaults location=westus2
```

## Create a storage account
We will use an Azure file share backed up by  Azure storage to store training data, training scripts, training logs and checkpoints, and the final model.
```
az storage account create --name <Storage Account Name> --sku Standard_LRS
```
### Set environmnent variables
To simplify further commands we can set up environmental variables with the Storage account name and the access key
```
az storage account keys list \
    -n <Storage account name> \
    -g <Resource group name> \
    -o table
export AZURE_STORAGE_ACCOUNT=<Storage account name>
export AZURE_STORAGE_ACCESS_KEY=<Storage account access key>
```

## Prepare Azure file share
### Create a file share
```
az storage share create \
    --account-name <Storage account Name> 
    --name <File share name>
```

### Create data and scripts directories in the share
```
az storage directory create \
    --share-name  <File share name>
    --name data
    
az storage directory create \
    --share-name  <File share name>
    --name scripts
```
### Copy training scripts
```
cd <Repo root>/Azure_AI_Infrastructure/labs/Lab01-SingleGPU
az storage file upload --share-name <File share name> --source train_eval.py --path scripts
az storage file upload --share-name <File share name> --source resnet.py --path scripts
```

### Copy training data
The training data in the TFRecords format have been uploaded to a public container in Azure storage. Use the following command to copy the files to your file share. The `--dryrun` option allows you to verify the configuration before starting the asynchronous copy operation.

```
az storage file copy start-batch \
  --destination-path data \
  --destination-share <File share name> \
  --source-account-name azaiworkshopst \
  --source-container tinyimagenet \
  --pattern '*' \
  --dryrun
```

### Verify that files are in the right folders
```
az storage file list --share-name <File share name> --path scripts -o table
az storage file list --share-name <File share name> --path data -o table
```

## Prepare a GPU cluster
### Create a single GPU VM node
```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuDSVM \
  --min 1 \
  --max 1 \
  --storage-account-name <Storage account name> \
  --afs-name <File share name> \
  --afs-mount-path external \
  --user-name <User name> \
  --passwor <Password>
```

It is recommended, although not required, to use ssh keys instead of passwords

```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuDSVM \
  --min 1 \
  --max 1 \
  --storage-account-name <Storage account name> \
  --afs-name <File share name> \
  --afs-mount-path external \
  --user-name $USER \
  -k ~/.ssh/id_rsa.pub
```
To generate `ssh` keys you can use an app of your choice including ssh-keygen:
```
ssh-keygen -t rsa
```

### Get cluster status
```
az batchai cluster list -o table
```

### List ssh connection info for the nodes in a cluster
```
az batchai cluster list-nodes -n <Cluster name> -g <Resource group name> -o table
```


## Create a training job

Create a JSON template file for job `job.json'`

### Create job
```
az batchai job create \
  --name <Job name> \
  --cluster-name <Cluster name> \
  --config job.json
```
### Monitor job
```
az batchai job list -o table
```

### List stdout and stderr output
```
az batchai job file list \
  --name <Job nme> \
  --resource-group <Resource group name> \
  --output-directory-id stdouterr
```

### Stream files from output directories
```
az batchai job file stream \
  -n <Job name> \
  -g <Resource group name> \
  -d stdouterr \
  -f <File to stream>
```

### Delete the job
```
az batchai job delete --name <Job name>
```

### Delete the  cluster
```
az batchai cluster delete --name <Cluster name>
```
## Using Tensorboard
### Mount a Tensorboard logdir folder
```
mkdir /mnt/jkbaidemofs
sudo mount -t cifs //jkbaidemost.file.core.windows.net/jkbaidemofs /mnt/jkbaidemofs -o vers=2.1, username=jkbaidemost,password=TDpMRqrllQEKO2Cw66s+nyIUD9hf5w0z1j8Rt2RIi2ROntbU3ta/o9zm8e+p7pllgSusWxazSDdbHEnXRjMIUg==,dir_mode=0777,file_mode=0777,serverino
```


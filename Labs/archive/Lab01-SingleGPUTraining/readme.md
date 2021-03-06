# Running a training job on a single GPU

The goal of this lab is to develop basic understanding of Azure Batch AI service and prepare Azure Batch AI environment for the labs focused on more advanced topics of distributed and parallel training.

**Follow the instructor**. The instructor will explain each step and deep dive into the algorithm used in the lab.

### Set environmnent variables
To simplify further commands we can set up environmental variables with the storage account name and the access key
```
az storage account keys list \
    -n <Storage account name> \
    -o table
export AZURE_STORAGE_ACCOUNT=<Storage account name>
export AZURE_STORAGE_ACCESS_KEY=<Storage account access key>
```

### Create a directory for the lab's scripts
```
az storage directory create \
    --share-name  <File share name>
    --name scripts/lab01
```

#### Copy training scripts
```
cd <Repo root>/Azure_AI_Infrastructure/Labs/Lab01-SingleGPU
az storage file upload --share-name <File share name> --source train_eval.py --path scripts/lab01
az storage file upload --share-name <File share name> --source resnet.py --path scripts/lab01
az storage file upload --share-name <File share name> --source feed.py --path scripts/lab01
```

#### Verify that files are in the right folders
```
az storage file list --share-name <File share name> --path scripts/lab01 -o table
az storage file list --share-name <File share name> --path data -o table
```

## Prepare a GPU cluster

```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuLTS \
  --min 3 \
  --max 3 \
  --storage-account-name <Storage account name> \
  --afs-name <File share name> \
  --afs-mount-path external \
  --user-name <User name> \
  --password <Password>
```

It is recommended, although not required, to use ssh keys instead of passwords

```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuLTS \
  --min 3 \
  --max 3 \
  --storage-account-name <Storage account name> \
  --afs-name <File share name> \
  --afs-mount-path external \
  --ssh-key ~/.ssh/id_rsa.pub \
  --user-name $USER 
  
```
To generate `ssh` keys you can use an app of your choice including ssh-keygen:
```
ssh-keygen -t rsa
```

Or you can generate ssh keys automatically during cluster creation
```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC6 \
  --image UbuntuLTS \
  --min 3 \
  --max 3 \
  --storage-account-name <Storage account name> \
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


## Create a training job

Walkthrough the job's python files and JSON template file for the job configuration `job.json'`

### Create job
```
az batchai job create \
  --name <Job name> \
  --cluster-name <Cluster name> \
  --config job.json
```
## Monitor the job
### List the jobs
```
az batchai job list -o table
```
### Show the job's status
```
az batchai job show -n <Job name>
```

### List stdout and stderr output
```
az batchai job file list \
  --name <Job nme> \
  --output-directory-id stdouterr
```

### Stream files from output directories
```
az batchai job file stream \
  -n <Job name> \
  -d stdouterr \
  -f <File to stream>
```
### Use Azure portal
You can also use Azure portal to monitor the job. 

### Use Tensorboard
#### Mount a Tensorboard logdir folder
##### On Linux
```
sudo mkdir /mnt/<Mount directory name>
sudo mount -t cifs //<Storage account name>.file.core.windows.net/<Share name> /mnt/<Mount directory name> -o vers=2.1,username=<Storage account name>,password=<Storage account key>,dir_mode=0777,file_mode=0777,serverino
```

Note: You may need to install CIFS utilities on your Linux machine. Azure DLVM and DSVM have these pre-installed:
```
sudo apt-get update
sudo apt-get install cifs-utils
```

##### On Windows
```
net use <Drive letter>: \\<Storage account name>.file.core.windows.net\<Share name> <Storage account key> /user:<Storage account name>
```
#### Start tensorboard
Start `tensorboard` on your development VM using the following command
```
tensorboard --logdir=<jobdir on a mount point> --ip=<IP address>
```


### Terminate/Delete the job
If you want to terminate or delet the job you can use the following commands
```
az batchai job terminate --name <Job name>
az batchai job delete --name <Job name>
```



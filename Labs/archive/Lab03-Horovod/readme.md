# Running a training job on a VM with multiple GPUs


### Set environmnent variables
To simplify further commands we can set up environmental variables with the storage account name and the access key
```
az storage account keys list \
    -n <Storage account name> \
    -o table
export AZURE_STORAGE_ACCOUNT=<Storage account name>
export AZURE_STORAGE_ACCESS_KEY=<Storage account access key>
```


### Copy training scripts
```
cd <Repo root>/Azure_AI_Infrastructure/labs/Lab03-Parallel-Horovod
az storage directory create --share-name  <File share name> --name scripts/lab03
az storage file upload --share-name <File share name> --source train_eval.py --path scripts/lab03
az storage file upload --share-name <File share name> --source resnet.py --path scripts/lab03
az storage file upload --share-name <File share name> --source feed.py --path scripts/lab03
```



### Verify that files are in the right folders
```
az storage file list --share-name <File share name> --path scripts/lab03 -o table
az storage file list --share-name <File share name> --path data -o table
```

## Prepare a GPU cluster

```
az batchai cluster create \
  --name  <Cluster name> \
  --vm-size STANDARD_NC12 \
  --image UbuntuDSVM \
  --min 1 \
  --max 1 \
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
sudo mount -t cifs //<Storage account name>.file.core.windows.net/jkbaidemofs /mnt/<Mount directory name> -o vers=2.1,username=<Storage account name>,password=<Storage account key>,dir_mode=0777,file_mode=0777,serverino
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


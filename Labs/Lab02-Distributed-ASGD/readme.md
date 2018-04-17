Running a training job on a cluster of GPUs



## Prepare a GPU cluster
### Create a three node GPU cluster

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
az batchai cluster list-nodes -n <Cluster name> -g <Resource group name> -o table
```

### Explore the cluster's node
```
ssh <IP address> -p node
cd /mnt/batch/tasks/shared/LS_root/mounts
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
#### On Linux
```
sudo mkdir /mnt/<Mount directory name>
sudo mount -t cifs //<Storage account name>.file.core.windows.net/jkbaidemofs /mnt/<Mount directory name> -o vers=2.1,username=<Storage account name> ,password=<Storage account key>,dir_mode=0777,file_mode=0777,serverino
```

Note: You may need to install CIFS utilities on your Linux machine. Azure DLVM and DSVM have these pre-installed:
```
sudo apt-get update
sudo apt-get install cifs-utils
```

#### On Windows
```
net use <Drive letter>: \\<Storage account name>.file.core.windows.net\<Share name> <Storage account key> /user:<Storage account name>
```




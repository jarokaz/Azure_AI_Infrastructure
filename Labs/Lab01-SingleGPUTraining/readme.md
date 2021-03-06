# Running a training job on a single GPU

The goal of this lab is to develop basic understanding of Azure Batch AI service and prepare Azure Batch AI environment for the labs focused on more advanced topics of distributed and parallel training.

**Follow the instructor**. The instructor will explain each step and deep dive into the algorithm used in the lab.


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
az storage file upload --share-name <File share name> --source model.py --path scripts/lab01
az storage file upload --share-name <File share name> --source feed.py --path scripts/lab01
```

#### Verify that files are in the right folders
```
az storage file list --share-name <File share name> --path scripts/lab01 -o table
az storage file list --share-name <File share name> --path data -o table
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



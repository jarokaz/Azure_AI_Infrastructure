# Running a training job on a cluster of GPUs using asynchronoues SGD in tensorflow


### Get cluster status
```
az batchai cluster list -o table
```
### List the jobs
```
az batchai job list -o table
```
Terminate/delete any jobs

### Upload python files from the lab to shared storage
```
az storage directory create --share-name <File share> --name scripts/lab02
az storage file upload --share-name <File share> --path scripts/lab02 --source train_eval.py
az storage file upload --share-name <File share> --path scripts/lab02 --source resnet.py
```

## Create a training job

Walk through python files and  `job.json'`

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
## Use Tensorboard 
Use the same process as in Lab01


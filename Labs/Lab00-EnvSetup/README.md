

# Workshop environment set up

To participate in the workshop you need a workstation with  `ssh`, `git`, text editor and the latest version of `Azure CLI 2.0`.

## Install pre-requisites on Windows 10

Although you can use any `ssh` and `git` clients, and you can install `Azure CLI` on Windows command prompt, we highly recommend using  Ubuntu on Windows Subsystem for Linux.

Install Windows Subsystem for Linux and Ubuntu distributions

https://docs.microsoft.com/en-us/windows/wsl/install-win10

Install `Azure CLI` for Ubuntu.

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-apt?view=azure-cli-latest

If you choose to use Window command prompt follow this instructions to install `Azure CLI`

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows?view=azure-cli-latest



## Install pre-requisites on Mac OS

Install Homebrew

https://docs.brew.sh/Installation.html

Install git
```
brew install git
```

Install `Azure CLI 2.0`

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-macos?view=azure-cli-latest

## Clone the workshop repo

Use the following command to clone the workshop repo:

``` 
git clone https://github.com/jarokaz/Azure_AI_Infrastructure.git
```

You can clone the repo into an arbitrary folder structure; however when using Windows Subsystem for Linux we highly recommend cloning the repo in the subfolder of a drive visible to Windows. For example:
```
/mnt/c/repos
```
This way you will be able to view/edit files using both Windows and Linux editors.

## Create the workshop's resource group and storage
All Azure resources created during the workshop will be hosted in the same resource group. It will simplify navigation and clean-up. This streamlined approach will work well for the workshop but does not represent the best practice for more complex production deployments. Refer to you organization's Azure guidance when setting up production grade environments.

The shared storage resources created during this labe will also be utilized by the following labs. 

### Login to your Azure subscription
```
az login
```
If you have multiple subscriptions set the right one with
```
az account set -s <Subscription ID>
```
### Register Batch AI resource providers
Make sure that Batch AI resource providers are registered for you subscription. This is a one-time configuration.
```
az provider register -n Microsoft.BatchAI
az provider register -n Microsoft.Batch
```
### Create a resource group

```
az group create --name <Resource group name> --location westus2
az configure --defaults group=<Resource group Name>
az configure --defaults location=westus2
```

### Create a storage account
We will use an Azure file share backed up by  Azure storage to store training data, training scripts, training logs, checkpoints, and the final model.
```
az storage account create --name <Storage Account Name> --sku Standard_LRS
```

```

### Prepare Azure file share
All labs in the workshop utilize Azure File Shares as shared storage. As noted by the instructor other shared storage options (e.g. NFS and distributed file systems) may perform better for really large data sets.

#### Create a file share
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









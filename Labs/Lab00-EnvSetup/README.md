# Workshop environment set up

## Provision Deep Learning Virtual Machine
Use Azure Portal to provision a Deep Learning Virtual Machine. Follow the instructor. Don't forget to change a DNS name.

## Configure ports for Tensorboard

![Tensorboard ports](images/tensorboard.jpg)
## Create Anaconda environment
Use ssh to logon to the virtual machine and create a new Anaconda environment
```
ssh <user name>@<vm name>.<region>.cloudapp.azure.com

conda create -n tf python=3.5 anaconda

source activate tf

```

## Clone this github site
```
git clone https://github.com/jarokaz/Azure_AI_Infrastructure.git
```

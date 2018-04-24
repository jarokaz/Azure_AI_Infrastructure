

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


You are now ready to move to Lab 1.







# Environment set up

## Set up walkthrough code
### Clone the github repo
```
cd
mkdir repos
cd repos
git clone  https://github.com/jarokaz/AzureAIInfrastructure.git
```
### Download training data
```
cd ~/repos/AzureAIInfrastructure
mkdir data
cd ~/repos/AzureAIInfrastructure/Labs/Lab00-DevEnv
python generate_cifar10_tfrecords.py
```
### Start a test run
```
cd ~/repos/AzureAIInfrastructure/Labs/Lab01-WarmUp
python train.py
```


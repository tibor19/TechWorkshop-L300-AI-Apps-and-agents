rgName=${1-"techworkshop-l300-ai-agents"}
location=${2-"swedencentral"}

az group create --name $rgName --location $location
az deployment group create --resource-group $rgName --template-file src/infra/DeployAzureResources.bicep --parameters userPrincipalId="$(az ad signed-in-user show --query id -o tsv)" -o tsv
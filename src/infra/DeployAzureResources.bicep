@description('Object ID of the user/principal to grant Cosmos DB data access')
// Get your principal object ID via: az ad signed-in-user show --query id -o tsv
param userPrincipalId string = deployer().objectId

@minLength(1)
@description('Primary location for all resources.')
param location string = resourceGroup().location

var projectPrefix = uniqueString(resourceGroup().id)
var cosmosDbName = '${projectPrefix}-cosmosdb'
var cosmosDbDatabaseName = 'zava'
var storageAccountName = '${projectPrefix}sa'
var aiFoundryName = 'aif-${projectPrefix}'
var aiProjectName = 'proj-${projectPrefix}'
var webAppName = '${projectPrefix}-app'
var appServicePlanName = '${projectPrefix}-cosu-asp'
var logAnalyticsName = '${projectPrefix}-cosu-la'
var appInsightsName = '${projectPrefix}-cosu-ai'
var webAppSku = 'S1'
var registryName = '${projectPrefix}cosureg'
var registrySku = 'Standard'

var tags = {
  Project: 'Tech Workshop L300 - AI Apps and Agents'
  Environment: 'Lab'
  ProjectPrefix: projectPrefix
  Owner: deployer().userPrincipalName
  SecurityControl: 'ignore'
  CostControl: 'ignore'
}

// Ensure the current resource group has the required tag via a subscription-scoped module
module updateRgTags 'updateRgTags.bicep' = {
  name: 'updateRgTags'
  scope: subscription()
  params: {
    rgName: resourceGroup().name
    rgLocation: resourceGroup().location
    newTags: union(resourceGroup().tags ?? {}, tags )
  }
}

var locations = [
  {
    locationName: location
    failoverPriority: 0
    isZoneRedundant: false
  }
]

@description('Creates an Azure Cosmos DB NoSQL account.')
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: cosmosDbName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  kind: 'GlobalDocumentDB'
  properties: {
    capabilities: [
      {
        name: 'EnableNoSQLVectorSearch'
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    databaseAccountOfferType: 'Standard'
    locations: locations
    disableLocalAuth: false
  }
  tags: tags
}

@description('Creates an Azure Cosmos DB NoSQL API database.')
resource cosmosDbDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosDbAccount
  name: cosmosDbDatabaseName
  properties: {
    resource: {
      id: cosmosDbDatabaseName
    }
  }
  tags: tags
}

@description('Creates an Azure Storage account.')
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowSharedKeyAccess: false
  }
  tags: tags
}

resource aiFoundry 'Microsoft.CognitiveServices/accounts@2025-10-01-preview' = {
  name: aiFoundryName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  kind: 'AIServices'
  properties: {
    // required to work in Microsoft Foundry
    allowProjectManagement: true 

    defaultProject: aiProjectName
    associatedProjects: [
      aiProjectName
    ]

    // Defines developer API endpoint subdomain
    customSubDomainName: aiFoundryName

    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}

/*
  Developer APIs are exposed via a project, which groups in- and outputs that relate to one use case, including files.
  Its advisable to create one project right away, so development teams can directly get started.
  Projects may be granted individual RBAC permissions and identities on top of what account provides.
*/ 
resource aiProject 'Microsoft.CognitiveServices/accounts/projects@2025-10-01-preview' = {
  name: aiProjectName
  parent: aiFoundry
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {}
  tags: tags
}

@description('Creates the gpt-5-mini model deployment.')
resource gpt5miniModel 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: aiFoundry
  name: 'gpt-5-mini'
  sku: {
    capacity: 10
    name: 'GlobalStandard'
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-5-mini'
      version: '2025-08-07'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    currentCapacity: 10
    raiPolicyName: 'Microsoft.DefaultV2'
  }
  dependsOn: [
    aiProject
  ]
}

@description('Creates the Phi-4 model deployment.')
resource phi4Model 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: aiFoundry
  name: 'Phi-4'
  sku: {
    capacity: 1
    name: 'GlobalStandard'
  }
  properties: {
    model: {
      name: 'Phi-4'
      format: 'Microsoft'
      version: '7'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    currentCapacity: 1
    raiPolicyName: 'Microsoft.DefaultV2'
  }
  dependsOn: [
    gpt5miniModel
  ]
}

@description('Creates the TextEmbedding3Large model deployment.')
resource modelTextEmbedding3Large 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: aiFoundry
  name: 'text-embedding-3-large'
  sku: {
    name: 'GlobalStandard'
    capacity: 425
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-large'
      version: '1'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    currentCapacity: 425
    raiPolicyName: 'Microsoft.DefaultV2'
  }
  dependsOn: [
    phi4Model
  ]
}



@description('Creates an Azure Log Analytics workspace.')
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
    workspaceCapping: {
      dailyQuotaGb: 1
    }
  }
  tags: tags
}

@description('Creates an Azure Application Insights resource.')
resource appInsights 'Microsoft.Insights/components@2020-02-02-preview' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
  tags: tags
}

@description('Creates an Azure Container Registry.')
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2022-12-01' = {
  name: registryName
  location: location
  sku: {
    name: registrySku
  }
  properties: {
    adminUserEnabled: false
  }
  tags: tags
}

@description('Creates an Azure App Service Plan.')
resource appServicePlan 'Microsoft.Web/serverFarms@2022-09-01' = {
  name: appServicePlanName
  location: location
  kind: 'linux'
  properties: {
    reserved: true
  }
  sku: {
    name: webAppSku
  }
  tags: tags
}

@description('Creates an Azure App Service for Zava.')
resource appServiceApp 'Microsoft.Web/sites@2022-09-01' = {
  name: webAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: 'DOCKER|${containerRegistry.name}${environment().suffixes.acrLoginServer}/${uniqueString(resourceGroup().id)}/techworkshopl300/zava:latest'
      http20Enabled: true
      minTlsVersion: '1.2'
      appCommandLine: ''
      appSettings: [{
          name: 'WEBSITES_ENABLE_APP_SERVICE_STORAGE'
          value: 'false'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${containerRegistry.name}${environment().suffixes.acrLoginServer}'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_USERNAME'
          value: containerRegistry.name
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: appInsights.properties.InstrumentationKey
      }]
    }
  }
  tags: tags
}

// Cosmos DB built-in data plane role IDs
// Reference: https://learn.microsoft.com/connectors/documentdb/#microsoft-entra-id-authentication-and-cosmos-db-connector
// var cosmosDbBuiltInDataReaderRoleId = '00000000-0000-0000-0000-000000000001'
var cosmosDbBuiltInDataContributorRoleId = '00000000-0000-0000-0000-000000000002'

// Azure RBAC role IDs
// Reference: https://learn.microsoft.com/azure/role-based-access-control/built-in-roles
// var cosmosDbAccountReaderRoleId = 'fbdf93bf-df7d-467e-a4d2-9458aa1360c8'
var cognitiveServicesOpenAIUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
var cognitiveServicesContributorRoleId = '25fbc0a9-bd7c-42a3-aa1a-3b75d497ee68'
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var acrPushRoleId = '8311e382-0749-4cb8-b61a-304f252e45ec'

@description('Assigns Cosmos DB Built-in Data Contributor role to the specified user')
resource cosmosDbDataContributorRoleAssignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2023-04-15' = {
  name: guid(cosmosDbAccount.id, userPrincipalId, cosmosDbBuiltInDataContributorRoleId)
  parent: cosmosDbAccount
  properties: {
    roleDefinitionId: '${cosmosDbAccount.id}/sqlRoleDefinitions/${cosmosDbBuiltInDataContributorRoleId}'
    principalId: userPrincipalId
    scope: cosmosDbAccount.id
  }
}

// Role assignments for Cosmos DB managed identity
@description('Assigns Cognitive Services OpenAI User role to Cosmos DB on AI Project')
resource cosmosDbProjectOpenAIUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aiProject.id, cosmosDbAccount.id, cognitiveServicesOpenAIUserRoleId)
  scope: aiProject
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: cosmosDbAccount.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Assigns Cognitive Services OpenAI User role to the user on Microsoft Foundry')
resource userFoundryOpenAIUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aiFoundry.id, userPrincipalId, cognitiveServicesOpenAIUserRoleId)
  scope: aiFoundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

@description('Assigns Cognitive Services OpenAI User role to Cosmos DB on Microsoft Foundry')
resource cosmosDbFoundryOpenAIUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aiFoundry.id, cosmosDbAccount.id, cognitiveServicesOpenAIUserRoleId)
  scope: aiFoundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: cosmosDbAccount.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Assigns Cognitive Services Contributor role to Cosmos DB on AI Project')
resource cosmosDbProjectContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aiProject.id, cosmosDbAccount.id, cognitiveServicesContributorRoleId)
  scope: aiProject
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesContributorRoleId)
    principalId: cosmosDbAccount.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Assigns StorageBlobDataContributor role to AI Project on storageAccount')
resource storageAccountProjectRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, aiProjectName, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributorRoleId)
    principalId: aiProject.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Assigns StorageBlobDataContributor role to user on storageAccount')
resource storageAccountUserRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, userPrincipalId, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributorRoleId)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

@description('Assigns ACR Pull role to the app service on Container Registry')
resource appServiceAcrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, appServiceApp.id, acrPullRoleId)
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: appServiceApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Role assignments for user
@description('Assigns ACR Push role to the user on Container Registry')
resource userAcrPushRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, userPrincipalId, acrPushRoleId)
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPushRoleId)
    principalId: userPrincipalId
    principalType: 'User'
  }
} 

output project_prefix string = projectPrefix
output cosmosDbEndpoint string = cosmosDbAccount.properties.documentEndpoint
output storageAccountName string = storageAccount.name
output container_registry_name string = containerRegistry.name
output application_name string = appServiceApp.name
output application_url string = appServiceApp.properties.hostNames[0]

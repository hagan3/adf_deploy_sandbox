# StandardAADeployTemplate

This terraform template creates the following resources in a new resource group:

- Storage Account
- Key Vault (Storage account key, sas token, SQL admin username, SQL admin password will be stored in the Key Vaults )
- Databricks Workspace
- Application Insights
- ML Workspace
- Azure SQL Server
- Azure Data Factory (Linked service to Storage will be created as well)
- Virtual Network with 3 subnets, first two subnets are for Databricks injected VNet, the third subnet is for the compute node for Azure Machine Workspace Studio to use.

## Deploy from local computer or cloudshell
Use AZ CLI to setup subscriptions to work with if run from local or cloudshell
```
# Login into Azure
az login

# Show subscription list that you have access to
az account list

# Setup the subscription the TerraForm to work with, replace [subscriptionname] with the subscription. e.g. SuncorABOSandbox
az account set --subscription [subscripitonname]

# Display the current subscription to confirm the subscription setup the step above
az account list --query "[?isDefault].{Name:name, Defaut:isDefault}" -o table
```

## Deploy using GitHub Action
There are two Workflows: Terraform Plan and Terraform Deploy. Terraform Plan is to plan the deployment, this should be run first, review of what to to be created is required to avoid mistake. Terraform Deploy is the actual deployment.

When deploy using GitHub Action, a form would need to be filled:
- Key: This is the key for GitHub to use to access Azure
- Prefix: This is the three character code used across Suncor, the location of the file is located in [LiveLink](http://ecm/ecmlivelinkprd/llisapi.dll?func=ll&objaction=overview&objid=122192557)
- Instance#: Default is 001, it is the sequence number of the resources to be created. 

## Post deployment activities
- By default, both storage account and SQL server can only be accessible from Suncor Calgary on-prem network and Azure service. If you need to open up for additional IP, please update firewall rule on Storage and SQL server. 
- Although Data Factory linked service for storage is created with configuration of using Managed Identity, but it would need to be reconfigured in the ADF as well. Please update the linked service to use the managed identity.

When creating compute node in Machine Learning Workspace, it needs to be created in the VNet, please choose the third subnet of the VNet in the same resource group. 

After executing the template you will need to configure and mount the storage account in the databricks environment.

1. Create a Databricks secret scope using the naming convention [keyvault_name]-scope-1. Go to https://<databricks-instance>#secrets/createScope. This URL is case sensitive; scope in createScope must be uppercase.

2. Create a single node cluster

3. Mount the storage account

Using Account Key:
```
dbutils.fs.mount(
   source = "wasbs://[container_name]@[storage_account_name].blob.core.windows.net",
   mount_point = "/mnt/[storage_account_name]/[container_name]",
   extra_configs = {
     "fs.azure.account.key.[storage_account_name].blob.core.windows.net": dbutils.secrets.get(scope = "[keyvault_name]-scope-1", key = "[storage_account_name]-key")
   }
 )
```
Using SAS Token:
```
dbutils.fs.mount(
   source = "wasbs://[container_name]@[storage_account_name].blob.core.windows.net",
   mount_point = "/mnt/[storage_account_name]/[container_name]",
   extra_configs = {
     "fs.azure.sas.[container_name].[storage_account_name].blob.core.windows.net": dbutils.secrets.get(scope = "[keyvault_name]-scope-1", key = "[storage_account_name]-token")
   }
 )
```

TODO: Automate databricks secret scope creation, cluster creation, and storage mount

## ReadMe for Machine Learning Industrialization Components

## main_adf_dev.tf
Main document for setting up azure data factory and data bricks terraform resources.

Resource Setup:
Configuring terraform and required providers, subscription ID, resource name and locations.

Storage/Data Factory:
- adfstorageal: Create standard-tier storageV2 account
- adfstoragesource01: configure private source storage container
- adfstoragetarget01: configure private target storage container
- adftestal: data factory hostname
- github configurations: account, branch, URL, repository, root folder variables all defined in the variables_adf_Dev.tf file
- key vault: includes secret configuration, token configuration, and SAS token

Linked Services:
- adfbloblink01: used for blob/datalake storage
- adflinkserviceweb: used for for CSV/Local Storage, HTTP, JSON, Parquet
- adflinksqlserver: used for Table/SQL Server Storage
- adflinkazurefilestorage: used for for Azure File Storage
- adflinksynapse: used for Synapse
- adflinkdatalakestoragegen2: used for for Data Lake Storage Gen2
- adflinkkeyvault: used for key vault

Datasets:
- adfdsblob01: dataset for blob/datalake storage
- adfdsdelimitedtextlink01: dataset for csv/local storage
- adfdsdatasource01: creating the datasource that is to be referenced by all dataflows
- adfdshttp01: HTTP dataset.  relative_url, request_body, request_method all need to be specified, temporary values currently in place
- adfdsjson01: JSON dataset.  relatve_url, path, filename all need to be specified, temporary values as currently in place
- adfdssqlserver01: dataset for table/SQL server storage

Dataflows:
All dataflows currently using Terraform ARM template.  
Use deployment mode 'Incremental' to create dataflows in ADF
- Aggregate Strings: After grouping by a specified column, aggregates all values from a field into a single string separated by commas. How to use: Populate the GroupByColumnName parameter to specify which column to group by and populate the AggregateColumnName parameter to combine the values into a single separated string. Update the source and sink reference datasets
- Calculate Moving Average: Calculates a 15 day moving average and rounds to two decimal places. How to use: Specify the KeyColumn to group by for example a stock ticker symbol. Specify the DateColumn to sort the dataset ascending. Specify the ValueColumn where the average will be performed on, this expects a double. Update the source and sink reference dataset.
- Check Null Columns: Identifies records where any column has a NULL value. How to use: Update the source and sink reference datasets
- Count Distinct Rows: Returns the first occurrence of all distinct rows based on all column values. How to use: Update the source and sink reference datasets
- Count Distinct Values: Aggregates the data to determine the number of distinct values for a specified field and the number of unique values for that field. How to use: Populate the ColumnName parameter to group by and the source and sink reference datasets
- Impute Fill Down: Replace NULL values with the values from the previous non-NULL value in the sequence. How to use: Update the SortByColumnName parameter with the column to sequence the dataset. Specify the FillColumnName to be imputed with the last known value. Update the source and sink reference datasets
- Persist Column Data Types: Generates the data type of each column to persist to a data store. How to use: Update the source and sink reference datasets.
- Remove Duplicates: Remove duplicates and returns the first instance for all records. How to use: Fill in the ColumnName Parameter with the field to group by and update the source and sink reference datasets.
- Summarize Data: Profiles the data by: Counting the number of nulls and non null records. Aggregating columns to calculate the stddev, min, max, avg and variance. Length of the columns which are type string. How to use: Update the source and sink reference datasets
- Remove Non Alphanumeric: Removes ^a-zA-Z\\d\\s: symbols from all fields

Pipelines:
- adfpipelinecopyfromblobtosqlserver: pipeline with the blob as source and SQL server as the sink, input is adfdsblob01, output is adfdssqlserver01
- adfpipelinecopyfromsqlservertoblob: pipeline to copy from SQL server as source to the blob as the sink, input is adfdssqlserver01, output is adfdsblob01
- adfpipelineremovealphanumeric: pipeline to copy data and execute dataflow to remove alphanumeric characters, blob source and SQL server sink
- adfpipelinecopynewfilesbylastmodified: pipeline to copy files using Azure Blob Storage read settings source, and sink using Azure Blob Write Settings
- adftriggerschedule: schedule to adjust frequency of pipeline execution, currently set to every 5 days

Databricks:
- resource-databricks-dev: Creates the Azure Databricks Workspace
- provider: Databricks provider should be updated to align with authentication standards 
- cluster-dev: Creates the Databricks Cluster with preloaded libraries
- ml_forecasting_notebook: Python notebooks for experimentation
- eda_notebook: Python notebooks for experimentation


## variables_adf_dev.tf
All variables are defined within this file including:
- resource group
- resource location
- subscription id
- resource prefix: currently defaulted to "suncor", update as needed
- resource instance: currently defaulted to "suncor", update as needed
- region
- github details
- SAS start and expire date
- common cost and tracking tags

## terraform.tfvars
Configures region to be Canada Central.
Sets the SAS start and expire dates, currently 2020-03-11 and 2028-03-11 respectively.
Defines common tags including consumer organization, environment type and support status.

## terraform.tfstate
- File automatically updated during every terraform plan and apply run

## terraform.tfstate.backup 
- File automatically updated during every terraform plan and apply run

## Archive folder
Holds currently unused files, including PROD version files.  Also holds previous variable declarations, which has been duplicated over into variables_adf_dev.tf

TODO: 
- Due to Terraform limitation, currently using ARM template resources to deploy dataflow components.  Needs to be updated to use Terraform dataflow resources when available
- Merge main_adf_dev.tf with main.tf, currently duplicated components have been commented out in main_adf_dev.tf 

# Main Azure Data Factory Terraform
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.46.0"
    }
	databricks = {
		source = "databrickslabs/databricks"
		version = "0.2.5"
	}
  }
}

# Configure Microsoft Azure Provider
provider "azurerm" {
  features {}
  subscription_id   = var.subscription-id
}


# Create resource group, name and location variables located in variables_adf_dev.tf
resource "azurerm_resource_group" "resource-group-dev" {
  name     = var.resource-group-dev
  location = var.resource-location
}

################################ Databricks ################################
/*
Code to create a new databricks workspace
*/
resource "azurerm_databricks_workspace" "resource-databricks-dev" {
  name                = "databricksdev"
  resource_group_name = var.resource-group-dev
  location            = var.resource-location
  sku                 = "trial"
}

provider "databricks" {
  #azure_workspace_resource_id = azurerm_databricks_workspace.resource-databricks-dev.id
  host = "https://adb-4569210438403194.14.azuredatabricks.net"
  token = "dapicc8de2a367c8f11be1aaaa3ce1d5423f"
}

/*
Cluster computing configuration to be updated as required.
*/
resource "databricks_cluster" "cluster-dev" {
  cluster_name            = "databricksclustersdev"
  spark_version           = "7.3.x-scala2.12"
  node_type_id            = "Standard_DS3_v2"
  autotermination_minutes = 20
  autoscale {
    min_workers = 1
    max_workers = 4
  }
  library {
    pypi {
        package = "scikit-learn==0.23.2"
        }
    }
  library {
    pypi {
        package = "adal==1.2.5"
        }
    }	
  library {
  	pypi {
  		package = "add-parent-path==0.1.11"
  		}
  	}
  library {
  	pypi {
  		package = "async-generator==1.10"
  		}
  	}
  library {
  	pypi {
  		package = "attrs==20.3.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-appconfiguration==1.1.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-batch==9.0.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-common==1.1.25"
  		}
  	}
  library {
  	pypi {
  		package = "azure-core==1.8.1"
  		}
  	}
  library {
  	pypi {
  		package = "azure-cosmos==3.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-datalake-store==0.0.50"
  		}
  	}
  library {
  	pypi {
  		package = "azure-functions-devops-build==0.0.22"
  		}
  	}
  library {
  	pypi {
  		package = "azure-graphrbac==0.61.1"
  		}
  	}
  library {
  	pypi {
  		package = "azure-identity==1.5.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-keyvault==4.1.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-keyvault-secrets==4.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-keyvault-administration==4.0.0b3"
  		}
  	}
  library {
  	pypi {
  		package = "azure-loganalytics==0.1.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-mgmt-datafactory==0.15.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-multiapi-storage==0.4.1"
  		}
  	}
  library {
  	pypi {
  		package = "azure-nspkg==3.0.2"
  		}
  	}
  library {
  	pypi {
  		package = "azure-storage-blob==12.5.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-storage-common==1.4.2"
  		}
  	}
  library {
  	pypi {
  		package = "azure-synapse-accesscontrol==0.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "azure-synapse-spark==0.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "backcall==0.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "bleach==3.2.1"
  		}
  	}
  library {
  	pypi {
  		package = "certifi==2020.12.5"
  		}
  	}
  library {
  	pypi {
  		package = "cffi==1.14.4"
  		}
  	}
  library {
  	pypi {
  		package = "chardet==3.0.4"
  		}
  	}
  library {
  	pypi {
  		package = "click==7.1.2"
  		}
  	}
  library {
  	pypi {
  		package = "conda-merge==0.1.5"
  		}
  	}
  library {
  	pypi {
  		package = "cryptography==3.3.1"
  		}
  	}
  library {
  	pypi {
  		package = "cycler==0.10.0"
  		}
  	}
  library {
  	pypi {
  		package = "databricks-cli==0.14.1"
  		}
  	}
  library {
  	pypi {
  		package = "decorator==4.4.2"
  		}
  	}
  library {
  	pypi {
  		package = "defusedxml==0.6.0"
  		}
  	}
  library {
  	pypi {
  		package = "entrypoints==0.3"
  		}
  	}
  library {
  	pypi {
  		package = "idna==2.10"
  		}
  	}
  library {
  	pypi {
  		package = "importlib-metadata==2.0.0"
  		}
  	}
  library {
  	pypi {
  		package = "iniconfig==1.1.1"
  		}
  	}
  library {
  	pypi {
  		package = "ipykernel==5.3.4"
  		}
  	}
  library {
  	pypi {
  		package = "ipython==7.18.1"
  		}
  	}
  library {
  	pypi {
  		package = "ipython-genutils==0.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "isodate==0.6.0"
  		}
  	}
  library {
  	pypi {
  		package = "jedi==0.17.2"
  		}
  	}
  library {
  	pypi {
  		package = "Jinja2==2.11.2"
  		}
  	}
  library {
  	pypi {
  		package = "jsonschema==3.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "MarkupSafe==1.1.1"
  		}
  	}
  library {
  	pypi {
  		package = "mistune==0.8.4"
  		}
  	}
  library {
  	pypi {
  		package = "msal==1.11.0"
  		}
  	}
  library {
  	pypi {
  		package = "msal-extensions~=0.3.0"
  		}
  	}
  library {
  	pypi {
  		package = "msrest==0.6.19"
  		}
  	}
  library {
  	pypi {
  		package = "msrestazure==0.6.4"
  		}
  	}
  library {
  	pypi {
  		package = "nbclient==0.5.1"
  		}
  	}
  library {
  	pypi {
  		package = "nbconvert==6.0.7"
  		}
  	}
  library {
  	pypi {
  		package = "nbformat==5.0.8"
  		}
  	}
  library {
  	pypi {
  		package = "nest-asyncio==1.4.3"
  		}
  	}
  library {
  	pypi {
  		package = "numpy==1.19.2"
  		}
  	}
  library {
  	pypi {
  		package = "oauthlib==3.1.0"
  		}
  	}
  library {
  	pypi {
  		package = "olefile==0.46"
  		}
  	}
  library {
  	pypi {
  		package = "packaging==20.7"
  		}
  	}
  library {
  	pypi {
  		package = "pandas==1.0.1"
  		}
  	}
  library {
  	pypi {
  		package = "pandocfilters==1.4.3"
  		}
  	}
  library {
  	pypi {
  		package = "parso==0.7.0"
  		}
  	}
  library {
  	pypi {
  		package = "pathlib2==2.3.5"
  		}
  	}
  library {
  	pypi {
  		package = "patsy==0.5.1"
  		}
  	}
  library {
  	pypi {
  		package = "pexpect==4.8.0"
  		}
  	}
  library {
  	pypi {
  		package = "pickleshare==0.7.5"
  		}
  	}
  library {
  	pypi {
  		package = "Pillow==8.1.0"
  		}
  	}
  library {
  	pypi {
  		package = "pip==21.1.1"
  		}
  	}
  library {
  	pypi {
  		package = "pluggy==0.13.1"
  		}
  	}
  library {
  	pypi {
  		package = "portalocker==1.7.1"
  		}
  	}
  library {
  	pypi {
  		package = "prompt-toolkit==3.0.7"
  		}
  	}
  library {
  	pypi {
  		package = "ptyprocess==0.6.0"
  		}
  	}
  library {
  	pypi {
  		package = "py==1.10.0"
  		}
  	}
  library {
  	pypi {
  		package = "pyarrow==3.0.0"
  		}
  	}
  library {
  	pypi {
  		package = "pycparser==2.20"
  		}
  	}
  library {
  	pypi {
  		package = "Pygments==2.7.1"
  		}
  	}
  library {
  	pypi {
  		package = "PyJWT==2.0.1"
  		}
  	}
  library {
  	pypi {
  		package = "pyodbc==4.0.30"
  		}
  	}
  library {
  	pypi {
  		package = "pyparsing==2.4.7"
  		}
  	}
  library {
  	pypi {
  		package = "pyrsistent==0.17.3"
  		}
  	}
  library {
  	pypi {
  		package = "pytest==6.2.1"
  		}
  	}
  library {
  	pypi {
  		package = "python-dateutil==2.8.1"
  		}
  	}
  library {
  	pypi {
  		package = "pytz==2020.5"
  		}
  	}
  library {
  	pypi {
  		package = "PyYAML==5.3.1"
  		}
  	}
  library {
  	pypi {
  		package = "pyzmq==19.0.2"
  		}
  	}
  library {
  	pypi {
  		package = "requests==2.25.0"
  		}
  	}
  library {
  	pypi {
  		package = "requests-oauthlib==1.3.0"
  		}
  	}
  library {
  	pypi {
  		package = "scipy==1.5.2"
  		}
  	}
  library {
  	pypi {
  		package = "six==1.15.0"
  		}
  	}
  library {
  	pypi {
  		package = "SQLAlchemy==1.3.19"
  		}
  	}
  library {
  	pypi {
  		package = "statsmodels==0.12.1"
  		}
  	}
  library {
  	pypi {
  		package = "stdlogging==0.16"
  		}
  	}
  library {
  	pypi {
  		package = "tabulate==0.8.7"
  		}
  	}
  library {
  	pypi {
  		package = "testpath==0.4.4"
  		}
  	}
  library {
  	pypi {
  		package = "toml==0.10.2"
  		}
  	}
  library {
  	pypi {
  		package = "tornado==6.0.4"
  		}
  	}
  library {
  	pypi {
  		package = "traitlets==5.0.4"
  		}
  	}
  library {
  	pypi {
  		package = "urllib3==1.26.2"
  		}
  	}
  library {
  	pypi {
  		package = "vsts==0.1.25"
  		}
  	}
  library {
  	pypi {
  		package = "wcwidth==0.2.5"
  		}
  	}
  library {
  	pypi {
  		package = "webencodings==0.5.1"
  		}
  	}
  library {
  	pypi {
  		package = "wheel==0.35.1"
  		}
  	}
  library {
  	pypi {
  		package = "xlrd==1.2.0"
  		}
  	}
  library {
  	pypi {
  		package = "zipp==3.4.0"
  		}
  	}
}

/* Code to add a user to databricks workspace
resource "databricks_scim_user" "admin" {
  user_name    = "admin@example.com"
  display_name = "Admin user"
  set_admin    = true
  default_roles = []
}
*/

resource "databricks_notebook" "notebook" {
  content = base64encode("print('Welcome to your Python notebook')")
  path = "/Shared/python_notebook"
  overwrite = false
  mkdirs = true
  language = "PYTHON"
  format = "SOURCE"
}

resource "databricks_notebook" "ml_forecasting_notebook" {
  content = base64encode(file("/Notebooks/auto-ml-forecasting-function.py"))
  path = "/Shared/auto-ml-forecasting-function"
  overwrite = false
  mkdirs = true
  language = "PYTHON"
  format = "SOURCE"
}

resource "databricks_notebook" "eda_notebook" {
  content = base64encode(file("/Notebooks/EDA.py"))
  path = "/Shared/EDA"
  overwrite = false
  mkdirs = true
  language = "PYTHON"
  format = "SOURCE"
}

/*
TO ADD
Additional Notebooks
Job
*/


################################ Storage/Data Factory ################################

# Create ADF storage account, constants defined in variables_adf_dev.tf file
resource "azurerm_storage_account" "adf_storage" {
  name                     = "adfstorageal"
  resource_group_name      = var.resource-group-dev
  location                 = var.resource-location
  account_tier             = "Standard"
  account_kind             = "StorageV2"
  account_replication_type = "LRS"
  tags = {
    environment = "dev"
    do          = "delete"
  }
}

# Configure storage container (source)
resource "azurerm_storage_container" "adf_storage_source_01" {
  name                  = "adfstoragesource01"
  storage_account_name  = azurerm_storage_account.adf_storage.name
  container_access_type = "private"
}

# Configure storage container (target)
resource "azurerm_storage_container" "adf_storage_target_01" {
  name                  = "adfstoragetarget01"
  storage_account_name  = azurerm_storage_account.adf_storage.name
  container_access_type = "private"
}

# Configure data factory
resource "azurerm_data_factory" "adf_test" {
  name                = "adftestal"
  resource_group_name = var.resource-group-dev
  location            = var.resource-location

  # Constants defined in variables file
  github_configuration {
    account_name    = var.github-account-name
    branch_name     = var.github-branch-name
    git_url         = var.github-git-url
    repository_name = var.github-repository-name
    root_folder     = var.github-root-folder
  }
}

# Configure key vault
resource "azurerm_key_vault" "vault" {
  name = "${var.resource_prefix}sbxarmkeyuw2${var.resource_instance}"
  resource_group_name = var.resource-group-dev
  location = var.resource-location
  tenant_id = data.azurerm_client_config.current.tenant_id
  sku_name = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "create",
      "get",
      "list",
      "update",
      "create",
      "import",
      "delete",
      "restore",
      "purge",
      "recover"
    ]

    secret_permissions = [
      "list",
      "set",
      "get",
      "delete",
      "purge",
      "recover",
      "backup",
      "restore"
    ]
  }

  #Access policy for AD Group RG-ARM-ANALYTICS-DE-CONTRIBUTOR: 
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = "2e5c5df3-18ab-46d9-8b68-5e8de931a8ff"

    key_permissions = [
      "create",
      "get",
      "list",
      "update",
      "create",
      "import",
      "delete",
      "restore",
      "purge",
      "recover"
    ]

    secret_permissions = [
      "list",
      "set",
      "get",
      "delete",
      "purge",
      "recover",
      "backup",
      "restore"
    ]
  }

  tags = var.common_tags
}

/*
# Configure key vault sercret
resource "azurerm_key_vault_secret" "secret" {
  name = "${azurerm_storage_account.adf_storage.name}-key"
  value = azurerm_storage_account.adf_storage.primary_access_key
  key_vault_id = azurerm_key_vault.vault.id
}
*/

# Get SAS token
data "azurerm_storage_account_sas" "storage_sas_token" {
  connection_string = azurerm_storage_account.adf_storage.primary_connection_string
  https_only        = true

  resource_types {
    service   = true
    container = true
    object    = true
  }

  services {
    blob  = true
    queue = false
    table = false
    file  = false
  }

  start  = var.sas_start
  expiry = var.sas_expire

  permissions {
    read    = true
    write   = true
    delete  = true
    list    = true
    add     = true
    create  = true
    update  = true
    process = true
  }
}

/*
# Store storage SAS token to key vault
resource "azurerm_key_vault_secret" "token" {
  name = "${azurerm_storage_account.adf_storage.name}-token"
  value = data.azurerm_storage_account_sas.storage_sas_token.sas
  key_vault_id = azurerm_key_vault.vault.id
  expiration_date = "${var.sas_expire}T23:59:59Z"
}
*/

################################ Linked Services ################################
##### Linked Service for Blob/Datalake Storage #####
resource "azurerm_data_factory_linked_service_azure_blob_storage" "adf_blob_link_01" {
  name                = "adfbloblink01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  connection_string   = azurerm_storage_account.adf_storage.primary_connection_string
}

##### Linked Service for CSV/Local Storage, HTTP, JSON, Parquet #####
resource "azurerm_data_factory_linked_service_web" "adf_link_service_web" {
  name                = "adflinkserviceweb"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  authentication_type = "Anonymous"
  url                 = "https://www.google.com"    # To be defined
}

##### Linked Service for Table/SQL Server Storage #####
resource "azurerm_data_factory_linked_service_sql_server" "adf_link_sql_server" {
  name                = "adflinksqlserver"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  connection_string   = "Integrated Security=False;Data Source=test;Initial Catalog=test;User ID=test;Password=test"
}

##### Linked Service for Azure File Storage #####
resource "azurerm_data_factory_linked_service_azure_file_storage" "adf_link_azure_file_storage" {
  name                = "adflinkazurefilestorage"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  connection_string   = azurerm_storage_account.adf_storage.primary_connection_string
}

##### Linked Service for Synapse #####
resource "azurerm_data_factory_linked_service_synapse" "adf_link_synapse" {
  name                = "adflinksynapse"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

  connection_string = "Integrated Security=False;Data Source=test;Initial Catalog=test;User ID=test;Password=test"
}

##### Data for Client Config of Data Lake Storage Gen2, Keyvault#####
data "azurerm_client_config" "current" {
}

##### Linked Service for Data Lake Storage Gen2 #####
resource "azurerm_data_factory_linked_service_data_lake_storage_gen2" "adf_link_data_lake_storage_gen2" {
  name                  = "adflinkdatalakestoragegen2"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  service_principal_id  = data.azurerm_client_config.current.client_id
  service_principal_key = "exampleKey"
  tenant                = "11111111-1111-1111-1111-111111111111"      # To be defined
  url                   = "https://datalakestoragegen2"               # To be defined
}

##### Linked Service for Key Vault #####
resource "azurerm_data_factory_linked_service_key_vault" "adf_link_key_vault" {
  name                = "adflinkkeyvault"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  key_vault_id        = azurerm_key_vault.vault.id
}


################################ ADF Datasets ################################

##### Dataset for Blob/Datalake Storage #####
resource "azurerm_data_factory_dataset_azure_blob" "adf_ds_blob_01" {
  name                = "adfdsblob01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_azure_blob_storage.adf_blob_link_01.name
  path                = azurerm_storage_container.adf_storage_source_01.name
}

##### Dataset for CSV/Local Storage #####
resource "azurerm_data_factory_dataset_delimited_text" "adf_ds_delimited_text_01" {
  name                = "adfdsdelimitedtextlink01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_web.adf_link_service_web.name

  http_server_location {
    relative_url = "http://www.google.com"    # To be defined
    path         = "foo/bar/"                 # To be defined
    filename     = "fizz.txt"                 # To be defined
  }

  column_delimiter    = ","
  row_delimiter       = "NEW"
  encoding            = "UTF-8"
  quote_character     = "x"
  escape_character    = "f"
  first_row_as_header = true
  null_value          = "NULL"
}

##### Datasource #####
resource "azurerm_data_factory_dataset_delimited_text" "adf_ds_datasource" {
  name                = "adfdsdatasource01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_web.adf_link_service_web.name

  http_server_location {
      relative_url = "http://www.google.com"    # To be defined
      path         = "foo/bar/"                 # To be defined
      filename     = "fizz.txt"                 # To be defined
    }

  column_delimiter    = ","
  row_delimiter       = "NEW"
  encoding            = "UTF-8"
  quote_character     = "x"
  escape_character    = "f"
  first_row_as_header = true
  null_value          = "NULL"
}

##### Dataset for HTTP #####
resource "azurerm_data_factory_dataset_http" "adf_ds_http_01" {
  name                = "adfdshttp01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_web.adf_link_service_web.name   # Uses linked service web (same as CSV)

  relative_url   = "http://www.google.com"    # To be defined
  request_body   = "foo=bar"                  # To be defined
  request_method = "POST"                     # To be defined
}

##### Dataset for JSON #####
resource "azurerm_data_factory_dataset_json" "adf_ds_json_01" {
  name                = "adfdsjson01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_web.adf_link_service_web.name

  http_server_location {
    relative_url = "/fizz/buzz/"            # To be defined
    path         = "foo/bar/"               # To be defined
    filename     = "foo.txt"                # To be defined
  }

  encoding = "UTF-8"
}

##### Dataset for Table/SQL Server Storage #####
resource "azurerm_data_factory_dataset_sql_server_table" "adf_ds_sql_server_01" {
  name                = "adfdssqlserver01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_sql_server.adf_link_sql_server.name
}

################################ ADF Dataflow ################################
# Resource: Dataset datasource
# Resource: Aggregate strings
# Resource: Calculate moving average
# Resource: Check null columns
# Resource: Count distinct rows
# Resource: Count distinct values
# Resource: Impute fill down
# Resource: Persist column data type
# Resource: Remove duplicates
# Resource: Summarize data
# Resource: Remove non-alphanumeric
# Descriptions for each resource outlined in properties/description of each dataflow

# Azure Terraform ARM Deployment templates
resource "azurerm_template_deployment" "adf_terraform_arm_deployment" {
  name                = "adfterraformarmdeployment"
  resource_group_name = var.resource-group-dev
  template_body = <<DEPLOY
{
  "$schema": "http://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",

  "contentVersion": "1.0.0.0",
  "parameters": {
    "factoryName": {
      "type": "string",
      "metadata": "Data Factory name",
      "defaultValue": "adftestal"
    }
  },
  "variables": {
        "factoryId": "[concat('Microsoft.DataFactory/factories/', parameters('factoryName'))]"
  },
  "resources": [
    {
      "name": "[concat(parameters('factoryName'), '/AggregateStrings')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "After grouping by a specified column, aggregates all values from a field into a single string separated by commas.\nHow to use:\nPopulate the GroupByColumnName parameter to specify which column to group by and populate the AggregateColumnName parameter to combine the values into a single separated string. Update the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "Aggregate1"
            },
            {
              "name": "StringAgg"
            }
          ],
          "script": "parameters{\n\tGroupByColumnName as string,\n\tAggregateColumnName as string\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 aggregate(groupBy(GroupByColumnName = $GroupByColumnName),\n\tstring_agg = collect($AggregateColumnName)) ~> Aggregate1\nAggregate1 derive(string_agg = toString(string_agg)) ~> StringAgg\nStringAgg sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/CalculateMovingAverage')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Calculates a 15 day moving average and rounds to two decimal places.\nHow to use:\n1. Specify the KeyColumn to group by for example a stock ticker symbol.\n2. Specify the DateColumn to sort the dataset ascending\n3. Specify the ValueColumn where the average will be performed on, this expects a double.\n4. Update the source and sink reference dataset.\n",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "Window1"
            }
          ],
          "script": "parameters{\n\tKeyColumn as string,\n\tDateColumn as date,\n\tValueColumn as double\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 window(over(MovingAverageColumn = $KeyColumn),\n\tasc($DateColumn, true),\n\tstartRowOffset: -7L,\n\tendRowOffset: 7L,\n\tFifteenDayMovingAvg = round(avg($ValueColumn),2)) ~> Window1\nWindow1 sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/CheckNullColumns')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Identifies records where any column has a NULL value.\nHow to use:\nUpdate the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "LookForNULLs"
            }
          ],
          "script": "source(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 split(contains(array(columns()),isNull(#item)),\n\tdisjoint: false) ~> LookForNULLs@(hasNULLs, noNULLs)\nLookForNULLs@hasNULLs sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/CountDistinctRows')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Returns the first occurrence of all distinct rows based on all column values.\nHow to use:\nUpdate the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "DistinctRows"
            }
          ],
          "script": "source(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 aggregate(groupBy(mycols = sha2(256,columns())),\n\teach(match(true()), $$ = first($$))) ~> DistinctRows\nDistinctRows sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/CountDistinctValues')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Aggregates the data to determine the number of distinct values for a specified field and the number of unique values for that field.\nHow to use:\nPopulate the ColumnName parameter to group by and the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "ValueDistAgg"
            },
            {
              "name": "UniqDist"
            }
          ],
          "script": "parameters{\n\tColumnName as string\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 aggregate(groupBy(GroupByFieldName = $ColumnName),\n\tcountunique = count()) ~> ValueDistAgg\nValueDistAgg aggregate(numofunique = countIf(countunique==1),\n\t\tnumofdistinct = countDistinct($ColumnName)) ~> UniqDist\nUniqDist sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/ImputeFillDown')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Replace NULL values with the values from the previous non-NULL value in the sequence.\nHow to use:\nUpdate the SortByColumnName parameter with the column to sequence the dataset.\nSpecify the FillColumnName to be imputed with the last known value.\nUpdate the source and sink reference datasets",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "DerivedColumn"
            },
            {
              "name": "SurrogateKey"
            },
            {
              "name": "Window1"
            },
            {
              "name": "Sort1"
            }
          ],
          "script": "parameters{\n\tFillColumnName as string,\n\tSortByColumnName as string\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nSort1 derive(dummy = 1) ~> DerivedColumn\nDerivedColumn keyGenerate(output(sk as long),\n\tstartAt: 1L) ~> SurrogateKey\nSurrogateKey window(over(dummy),\n\tasc(sk, true),\n\tFilledColumn = coalesce($FillColumnName, last($FillColumnName, true()))) ~> Window1\nsource1 sort(asc($SortByColumnName, true)) ~> Sort1\nWindow1 sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/PersistColumnDataTypes')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Generates the data type of each column to persist to a data store.\nHow to use:\nUpdate the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "DerivedColumn1"
            }
          ],
          "script": "source(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 derive(each(match(type=='string'), $$ = 'string'),\n\t\teach(match(type=='integer'), $$ = 'integer'),\n\t\teach(match(type=='short'), $$ = 'short'),\n\t\teach(match(type=='complex'), $$ = 'complex'),\n\t\teach(match(type=='array'), $$ = 'array'),\n\t\teach(match(type=='float'), $$ = 'float'),\n\t\teach(match(type=='date'), $$ = 'date'),\n\t\teach(match(type=='timestamp'), $$ = 'timestamp'),\n\t\teach(match(type=='boolean'), $$ = 'boolean'),\n\t\teach(match(type=='long'), $$ = 'long'),\n\t\teach(match(type=='double'), $$ = 'double')) ~> DerivedColumn1\nDerivedColumn1 sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/RemoveDuplicates')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Remove duplicates and returns the first instance for all records\nHow to use:\nFill in the ColumnName Parameter with the field to group by and update the source and sink reference datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "RemoveDuplicates"
            }
          ],
          "script": "parameters{\n\tColumnName as string\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 aggregate(groupBy(GroupByFieldName = $ColumnName),\n\teach(match(name!=$ColumnName), $$ = first($$))) ~> RemoveDuplicates\nRemoveDuplicates sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/SummarizeData')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Profiles the data by: \n1. Counting the number of nulls and non null records\n2. Aggregating columns to calculate the stddev, min, max, avg and variance\n3. Length of the columns which are type string\nHow to use:\nUpdate the source and sink reference datasets",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            }
          ],
          "transformations": [
            {
              "name": "SummaryStats"
            }
          ],
          "script": "source(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource1 aggregate(each(match(true()), $$+'_NotNull' = countIf(!isNull($$)), $$ + '_Null' = countIf(isNull($$))),\n\t\teach(match(type=='double'||type=='integer'||type=='short'||type=='decimal'), $$+'_stddev' = round(stddev($$),2), $$ + '_min' = min ($$), $$ + '_max' = max($$), $$ + '_average' = round(avg($$),2), $$ + '_variance' = round(variance($$),2)),\n\t\teach(match(type=='string'), $$+'_maxLength' = max(length($$)))) ~> SummaryStats\nSummaryStats sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1"
        }
      }
    },
    {
      "name": "[concat(parameters('factoryName'), '/RemoveNonAlphanumeric')]",
      "type": "Microsoft.DataFactory/factories/dataflows",
      "apiVersion": "2018-06-01",
      "properties": {
        "description": "Removes ^a-zA-Z\\d\\s: symbols from all fields\nHow to use:\nEither apply the transformation on all columns in the dataset or provide the column name to be transform through the ColumnName parameter. Update the source and sink datasets.",
        "type": "MappingDataFlow",
        "typeProperties": {
          "sources": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source1"
            },
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "source2"
            }
          ],
          "sinks": [
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink1"
            },
            {
              "dataset": {
                "referenceName": "adfdsdatasource01",
                "type": "DatasetReference"
              },
              "name": "sink2"
            }
          ],
          "transformations": [
            {
              "name": "AllColumns"
            },
            {
              "name": "OneColumn"
            }
          ],
          "script": "parameters{\n\tColumnName as string\n}\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source1\nsource(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tignoreNoFilesFound: false) ~> source2\nsource1 derive(each(match(true()), $$ = regexReplace($$,`^a-zA-Z\\d\\s:`,''))) ~> AllColumns\nsource2 derive(each(match(name==$ColumnName), $$ = regexReplace($$,`^a-zA-Z\\d\\s:`,''))) ~> OneColumn\nAllColumns sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink1\nOneColumn sink(allowSchemaDrift: true,\n\tvalidateSchema: false,\n\tskipDuplicateMapInputs: true,\n\tskipDuplicateMapOutputs: true) ~> sink2"
        }
      }
    }
]
}
DEPLOY

deployment_mode = "Incremental"
}

################################ Pipelines with JSON ################################

##### Pipeline to copy from Blob to SQL server #####
resource "azurerm_data_factory_pipeline" "adf_pipeline_copy_from_blob_to_sql_server" {
  name                = "adfpipelinecopyfromblobtosqlserver"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

activities_json = <<JSON
[
  {
      "name": "Copy_BlobToSqlServer",
      "type": "Copy",
      "dependsOn": [],
      "policy": {
          "timeout": "7.00:00:00",
          "retry": 0,
          "retryIntervalInSeconds": 30,
          "secureOutput": false,
          "secureInput": false
      },
      "userProperties": [],
      "typeProperties": {
          "source": {
              "type": "BlobSource",
              "recursive": true
          },
          "sink": {
              "type": "SqlServerSink",
              "tableOption": "autoCreate"
          },
          "enableStaging": false
      },
      "inputs": [
          {
              "referenceName": "adfdsblob01",
              "type": "DatasetReference"
          }
      ],
      "outputs": [
          {
              "referenceName": "adfdssqlserver01",
              "type": "DatasetReference"
          }
      ]
  }
]
JSON
}

##### Pipeline to copy from SQL server to Blob #####
resource "azurerm_data_factory_pipeline" "adf_pipeline_copy_from_sql_server_to_blob" {
  name                = "adfpipelinecopyfromsqlservertoblob"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

activities_json = <<JSON
[
  {
    "name": "Copy_SqlServerToBlob",
    "type": "Copy",
    "dependsOn": [],
    "policy": {
        "timeout": "7.00:00:00",
        "retry": 0,
        "retryIntervalInSeconds": 30,
        "secureOutput": false,
        "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
        "source": {
            "type": "SqlServerSource",
            "queryTimeout": "02:00:00",
            "partitionOption": "None"
        },
        "sink": {
            "type": "BlobSink"
        },
        "enableStaging": false
    },
    "inputs": [
        {
            "referenceName": "adfdssqlserver01",
            "type": "DatasetReference"
        }
    ],
    "outputs": [
        {
            "referenceName": "adfdsblob01",
            "type": "DatasetReference"
        }
    ]
}
]
JSON
}

##### Pipeline to Copy Data and Execute Dataflow to Remove Alphanumeric Characters #####
resource "azurerm_data_factory_pipeline" "adf_pipeline_remove_alphanumeric" {
  name                = "adfpipelineremovealphanumeric"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

activities_json = <<JSON
[
  {
    "name": "Copy data1",
    "type": "Copy",
    "dependsOn": [],
    "policy": {
      "timeout": "7.00:00:00",
      "retry": 0,
      "retryIntervalInSeconds": 30,
      "secureOutput": false,
      "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
        "source": {
          "type": "BlobSource",
          "recursive": true
        },
        "sink": {
          "type": "SqlServerSink"
        },
        "enableStaging": false
    },
    "inputs": [
        {
          "referenceName": "adfdsblob01",
          "type": "DatasetReference"
        }
    ],
    "outputs": [
        {
          "referenceName": "adfdssqlserver01",
          "type": "DatasetReference"
        }
    ]
  },
  {
    "name": "Dataflow",
    "type": "ExecuteDataFlow",
    "dependsOn": [
        {
            "activity": "Copy data1",
            "dependencyConditions": [
                "Succeeded"
            ]
        }
    ],
    "policy": {
      "timeout": "1.00:00:00",
      "retry": 0,
      "retryIntervalInSeconds": 30,
      "secureOutput": false,
      "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
      "dataflow": {
          "referenceName": "RemoveNonAlphanumeric",
          "type": "DataFlowReference"
      },
      "compute": {
          "coreCount": 8,
          "computeType": "General"
      },
      "traceLevel": "Fine"
    }
  }
]
JSON
}

###### Trigger Schedule #####
resource "azurerm_data_factory_trigger_schedule" "adf_trigger_schedule" {
  name                = "adftriggerschedule"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  pipeline_name       = azurerm_data_factory_pipeline.adf_pipeline_copy_from_blob_to_sql_server.name

  interval  = 5
  frequency = "Day"
}

################################## UNUSED #####################################
/*
##### Dataset for Parquet #####
# Requires more detailed connections, linked_service_name, also needs updated http_server_location details
resource "azurerm_data_factory_dataset_parquet" "adf_ds_parquet_01" {
  name                = "adfdsparquet01"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  linked_service_name = azurerm_data_factory_linked_service_web.adf_link_service_web.name

  http_server_location {
    relative_url = "http://www.google.com"
    path         = "foo/bar/"
    filename     = "fizz.txt"
  }
}
*/

/*
##### Data for Cosmos DB Storage #####
data "azurerm_cosmosdb_account" "adf_data_cosmosdb_account" {
  name                = "tfex-cosmosdb-account"
  resource_group_name = "tfex-cosmosdb-account-rg"
}

##### Linked Service for Cosmos DB Storage #####
# Requires more resource details, account_endpoint, account_key, database
resource "azurerm_data_factory_linked_service_cosmosdb" "adf_link_cosmosdb" {
  name                = "adflinkcosmosdb"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  account_endpoint    = azurerm_cosmosdb_account.adf_data_cosmosdb_account.endpoint
  account_key         = azurerm_cosmosdb_account.adf_data_cosmosdb_account.primary_access_key
  database            = "foo"
}
*/

/*
##### Data for Function App #####
data "azurerm_function_app" "adf_data_function_app" {
  name                = "test-azure-functions"
  resource_group_name = var.resource-group-dev
}

##### Linked Service for Azure Function #####
# Requires resource details, including URL and key
resource "azurerm_data_factory_linked_service_azure_function" "adf_link_azure_function" {
  name                = "adflinkazurefunction"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name
  url                 = "https://${data.azurerm_function_app.adf_data_function_app.default_hostname}"
  key                 = "foo"
}
*/

/*
################################ Databricks ################################
# Unused databricks code as more resource details are needed, connection details need to be updated

# create pub subnet for databricks hosts
resource "azurerm_subnet" "pub_subnet" {
  name                 = "${var.resource_prefix}sbxarmsubuw2001"
  virtual_network_name = azurerm_virtual_network.injected_vnet.name
  resource_group_name  = azurerm_resource_group.rg.name
  address_prefixes     = ["172.17.0.0/26"]
  service_endpoints    = ["Microsoft.Storage"]
  delegation {
    name = "delegation"
    service_delegation {
      name = "Microsoft.Databricks/workspaces"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action","Microsoft.Network/virtualNetworks/subnets/prepareNetworkPolicies/action","Microsoft.Network/virtualNetworks/subnets/unprepareNetworkPolicies/action"]
    }
  }
}

# create private subnet for databrics container
resource "azurerm_subnet" "pri_subnet" {
  name                 = "${var.resource_prefix}sbxarmsubuw2002"
  virtual_network_name = azurerm_virtual_network.injected_vnet.name
  resource_group_name  = azurerm_resource_group.rg.name
  address_prefixes     = ["172.17.0.64/26"]
  service_endpoints    = ["Microsoft.Storage"]
  delegation  {
    name = "delegation"
    service_delegation {
      name = "Microsoft.Databricks/workspaces"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action","Microsoft.Network/virtualNetworks/subnets/prepareNetworkPolicies/action","Microsoft.Network/virtualNetworks/subnets/unprepareNetworkPolicies/action"]
    }
  }
}

resource "azurerm_databricks_workspace" "databricks" {
  name = "${var.resource_prefix}sbxarmadbuw2${var.resource_instance}"
  resource_group_name = azurerm_resource_group.rg.name
  location = var.region
  sku = "standard"
  managed_resource_group_name="databricks-rg-${var.resource_prefix}sbxarmadbuw2${var.resource_instance}"

  custom_parameters {
    no_public_ip = true
    public_subnet_name = azurerm_subnet.pub_subnet.name
    private_subnet_name = azurerm_subnet.pri_subnet.name
    virtual_network_id = azurerm_virtual_network.injected_vnet.id
  }
  tags = var.common_tags
}
*/

/*
################################ Azure Machine Learning ################################
# Unused databricks code as more resource details are needed, connection details need to be updated

resource "azurerm_resource_group" "rg" {
  name = "${var.resource_prefix}sbxarmrgp${var.resource_instance}"
  location = var.region

  tags = var.common_tags
}

# Create VNet
resource "azurerm_virtual_network" "injected_vnet" {
  name = "${var.resource_prefix}sbxarmvntuw2${var.resource_instance}"
  resource_group_name = azurerm_resource_group.rg.name
  location = var.region
  address_space       = ["172.17.0.0/23"]
  tags = var.common_tags
}

# Create  subnet machine learning worksplace compute
resource "azurerm_subnet" "aml_subnet" {
  name                 = "${var.resource_prefix}sbxarmsubuw2003"
  virtual_network_name = azurerm_virtual_network.injected_vnet.name
  resource_group_name  = azurerm_resource_group.rg.name
  address_prefixes     = ["172.17.0.128/26"]
  service_endpoints    = ["Microsoft.Storage"]
}

resource "azurerm_machine_learning_workspace" "aml" {
  name = "${var.resource_prefix}sbxarmmlwuw2${var.resource_instance}"
  location = var.region
  resource_group_name = azurerm_resource_group.rg.name
  key_vault_id = azurerm_key_vault.vault.id
  storage_account_id = azurerm_storage_account.storage.id
  application_insights_id = azurerm_application_insights.insights.id
  identity {
    type = "SystemAssigned"
  }

  tags = var.common_tags
}
*/

/*
##### Pipeline to Bulk Copy from Files to Database #####
# Requires revisions to ForEach section of JSON template

resource "azurerm_data_factory_pipeline" "adf_pipeline_04" {
  name                = "adfpipelinebulkcopyfilestodatabase"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

# Current configurations: GetMetadataDataset and SourceDataset are set to adflinkdatalakestoragegen2, and SinkDataset set to adflinksynapse
activities_json = <<JSON
[
  {
    "name": "Get Files",
    "type": "GetMetadata",
    "dependsOn": [],
    "policy": {
        "timeout": "7.00:00:00",
        "retry": 0,
        "retryIntervalInSeconds": 30,
        "secureOutput": false,
        "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
        "dataset": {
            
            "type": "DatasetReference",
            "parameters": {
                "SourceContainer": {
                    "value": "",
                    "type": "Expression"
                },
                "SourceDirectory": {
                    "value": "",
                    "type": "Expression"
                }
            }
        },
        "fieldList": [
            "childItems"
        ],
        "storeSettings": {
            "type": "AzureBlobFSReadSettings",
            "recursive": true
        },
        "formatSettings": {
            "type": "DelimitedTextReadSettings"
        }
    }
  },
  {
      "name": "ForEachFile",
      "type": "ForEach",
      "dependsOn": [
          {
              "activity": "Get Files",
              "dependencyConditions": [
                  "Succeeded"
              ]
          }
      ],
      "userProperties": [],
      "typeProperties": {
          "items": {
              "value": "",
              "type": "Expression"
          },
          "activities": [
              {
                  "name": "CopyFiles",
                  "type": "Copy",
                  "dependsOn": [],
                  "policy": {
                      "timeout": "7.00:00:00",
                      "retry": 0,
                      "retryIntervalInSeconds": 30,
                      "secureOutput": false,
                      "secureInput": false
                  },
                  "userProperties": [],
                  "typeProperties": {
                      "source": {
                          "type": "DelimitedTextSource",
                          "storeSettings": {
                              "type": "AzureBlobFSReadSettings",
                              "recursive": true
                          },
                          "formatSettings": {
                              "type": "DelimitedTextReadSettings"
                          }
                      },
                      "sink": {
                          "type": "SqlDWSink",
                          "allowPolyBase": true,
                          "polyBaseSettings": {
                              "rejectValue": 0,
                              "rejectType": "value",
                              "useTypeDefault": true
                          }
                      },
                      "enableStaging": true
                  },
                  "inputs": [
                      {
                          "type": "DatasetReference",
                          "parameters": {
                              "fileName": {
                                  "value": "",
                                  "type": "Expression"
                              },
                              "SourceContainer": {
                                  "value": "",
                                  "type": "Expression"
                              },
                              "SourceDirectory": {
                                  "value": "",
                                  "type": "Expression"
                              }
                          }
                      }
                  ],
                  "outputs": [
                      {
                          "type": "DatasetReference",
                          "parameters": {
                              "sinkTableName": {
                                  "value": "",
                                  "type": "Expression"
                              }
                          }
                      }
                  ]
              }
          ]
      }
  }
]
JSON
}
*/

/*
##### Pipeline to Move Files #####
# Requires revisions to ForEach section of JSON template

resource "azurerm_data_factory_pipeline" "adf_pipeline_06" {
  name                = "adfpipelinecopysaptoadls"
  resource_group_name = var.resource-group-dev
  data_factory_name   = azurerm_data_factory.adf_test.name

activities_json = <<JSON
[
  {
    "name": "GetFileList",
    "description": "Get the list of file",
    "type": "GetMetadata",
    "dependsOn": [],
    "policy": {
        "timeout": "7.00:00:00",
        "retry": 0,
        "retryIntervalInSeconds": 30,
        "secureOutput": false,
        "secureInput": false
    },
    "userProperties": [],
    "typeProperties": {
        "dataset": {
            "type": "DatasetReference",
            "parameters": {
                "Container": {
                    "value": "",
                    "type": "Expression"
                },
                "Directory": {
                    "value": "",
                    "type": "Expression"
                }
            }
        },
        "fieldList": [
            "childItems"
        ],
        "storeSettings": {
            "type": "AzureBlobStorageReadSettings",
            "recursive": true
        }
    }
  },
  {
    "name": "ForEachFile",
    "description": "Iterate each file, and move them one by one.",
    "type": "ForEach",
    "dependsOn": [
        {
            "activity": "FilterFiles",
            "dependencyConditions": [
                "Succeeded"
            ]
        }
    ],
    "userProperties": [],
    "typeProperties": {
        "items": {
            "value": "",
            "type": "Expression"
        },
        "batchCount": 20,
        "activities": [
            {
                "name": "CopyAFile",
                "description": "Copy a file from the source store to the destination store.",
                "type": "Copy",
                "dependsOn": [],
                "policy": {
                    "timeout": "7.00:00:00",
                    "retry": 0,
                    "retryIntervalInSeconds": 30,
                    "secureOutput": false,
                    "secureInput": false
                },
                "userProperties": [],
                "typeProperties": {
                    "source": {
                        "type": "BinarySource",
                        "storeSettings": {
                            "type": "AzureBlobStorageReadSettings",
                            "recursive": true
                        },
                        "recursive": false
                    },
                    "sink": {
                        "type": "BinarySink",
                        "storeSettings": {
                            "type": "AzureBlobFSWriteSettings"
                        }
                    },
                    "enableStaging": false,
                    "dataIntegrationUnits": 0
                }
            },
            {
                "name": "DeleteAFile",
                "description": "Delete a file from the source store",
                "type": "Delete",
                "dependsOn": [
                    {
                        "activity": "CopyAFile",
                        "dependencyConditions": [
                            "Succeeded"
                        ]
                    }
                ],
                "policy": {
                    "timeout": "7.00:00:00",
                    "retry": 0,
                    "retryIntervalInSeconds": 30,
                    "secureOutput": false,
                    "secureInput": false
                },
                "userProperties": [],
                "typeProperties": {
                    "dataset": {
                        "type": "DatasetReference",
                        "parameters": {
                            "Container": {
                                "value": "",
                                "type": "Expression"
                            },
                            "Directory": {
                                "value": "",
                                "type": "Expression"
                            },
                            "filename": {
                                "value": "",
                                "type": "Expression"
                            }
                        }
                    },
                    "enableLogging": false,
                    "storeSettings": {
                        "type": "AzureBlobStorageReadSettings",
                        "recursive": true
                    }
                }
            }
        ]
    }
  },
  {
    "name": "FilterFiles",
    "description": "Only files will be selected, the source-folders will not be selected.",
    "type": "Filter",
    "dependsOn": [
        {
            "activity": "GetFileList",
            "dependencyConditions": [
                "Succeeded"
            ]
        }
    ],
    "userProperties": [],
    "typeProperties": {
        "items": {
            "value": "",
            "type": "Expression"
        },
        "condition": {
            "value": "",
            "type": "Expression"
        }
    }
  }
]
JSON
}
*/

Run the command example: python scryfall.py inside the Randall_Noval folder on the command line
must be installed python3+, openpyxl(pip install openpyxl), requests(pip install requests), pandas(pip install pandas),base64(pip install base64)
You have to edit config.py:
csv_file = 'cards-sample.csv' # csv file name
github_username = 'randynov' # your github user_name
github_token = ''	# your github access token (login github->settings->Developer settings->personal access token->generate new token you can slect all scopes and generate token
github_repo_name = 'mtgmanager2aetherhub' # target repo name
Written using the pep-8 standard
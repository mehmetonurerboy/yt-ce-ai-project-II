Scope of the project, conda is used as environment. So, all needy library is listed on environment.yml file

With this command on conda, you can create your environment :
> conda env create -f environment.yml

After this creation operation, you have to activate this environment for running codes. With this commands you can do it:
> conda activate credit_card

The kaggle dataset is must be downloaded for this tries. The downloaded file is accepted with this name : creditcard.csv

For preparing the train and test datasets, enter this command at this codes' directory :
> python prepare_datasets.py creditcard.csv

For running naive bayes :
> python naive_bayes.py

For kNN running :
> python knn.py

For LVQ running :
> python lvq.py

Attention : It is This python commands should run sequentially.
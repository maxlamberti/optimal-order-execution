# Optimal Order Execution
Exploring optimal order execution policies on limit order books

### Downloading Data From S3

Execute the script ```scripts/scrapers/download_s3_data.py``` with command line arguments like:

```python download_s3_data.py --pair=XBTZUSD --bucket=my-bucket --save_directory=/path/to/save/location --access_key_id=KEY_ID --secret_access_key=SECRET_KEY```

Valid parameters for pair argument: {ALL, USDTZUSD, XETHZUSD, XXMRZUSD, XXRPZUSD, XREPZUSD, XBTZUSD}


### Loading The Data Into Python

Check ```scripts/utils/data_loading.py``` for the ```load_data(...)``` method. You can specify the directory containing the parquet data and the method will combine the separate files into a single data frame. In case of the limit order book data, the whole data set might not fit into memory. In that case, datetime parameters can be passed to only select the subset of data scraped within those dates. Exact instructions are in the docstring.

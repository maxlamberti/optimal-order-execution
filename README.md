# Optimal Order Execution
Exploring optimal order execution policies on limit order books

### Downloading Data From S3

Execute the script ```scripts/scrapers/download_s3_data.py``` with command line arguments like:

```python download_s3_data.py --pair=XBTZUSD --bucket=my-bucket --save_directory=/path/to/save/location```

Valid parameters for pair argument: {ALL, USDTZUSD, XETHZUSD, XXMRZUSD, XXRPZUSD, XREPZUSD, XBTZUSD}

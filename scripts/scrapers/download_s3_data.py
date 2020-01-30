import os
import boto3
import logging
import argparse
from tqdm import tqdm


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s', level=logging.INFO)


def download_folder_from_s3(bucket, bucket_folder, save_directory=None, access_key_id=None, secret_access_key=None):
	"""Download data to from S3 bucket."""

	# define aws resource
	if (access_key_id is not None) and (secret_access_key is not None):
		logging.info("Initializing AWS session with custom credentials.")
		session = boto3.Session(
			aws_access_key_id=args.access_key_id,
			aws_secret_access_key=args.secret_access_key
		)
		s3_resource = session.resource('s3')
	else:
		s3_resource = boto3.resource('s3')

	# access bucket
	bucket = s3_resource.Bucket(bucket)
	logging.info("Starting download...")
	for object in tqdm(bucket.objects.filter(Prefix=bucket_folder)):

		# create save directory if doesnt exist
		save_filename = os.path.join(save_directory, object.key)
		if not os.path.exists(os.path.dirname(save_filename)):
			os.makedirs(os.path.dirname(save_filename))

		# download
		bucket.download_file(object.key, save_filename)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-b',
		'--bucket',
		type=str,
		required=True,
		help='Name of S3 bucket to download data from.'
	)
	parser.add_argument(
		'-s',
		'--save_directory',
		type=str,
		required=True,
		help='Directory path to save the file to.'
	)
	parser.add_argument(
		'-p',
		'--pair',
		type=str,
		default='ALL',
		required=False,
		help='Which currency pair to download data for. Set to ALL to get everything.'
	)
	parser.add_argument(
		'-i',
		'--access_key_id',
		type=str,
		default=None,
		required=False,
		help='Access key ID of AWS credentials.'
	)
	parser.add_argument(
		'-k',
		'--secret_access_key',
		type=str,
		default=None,
		required=False,
		help='Secret access key of AWS credentials.'
	)
	args = parser.parse_args()

	# choose currencies to retrieve
	if args.pair.upper() == 'ALL':
		currencies = ['USDTZUSD', 'XETHZUSD', 'XXMRZUSD', 'XXRPZUSD', 'XREPZUSD', 'XXBTZUSD']
	else:
		currencies = [args.pair]

	for pair in currencies:
		logging.info("Downloading data for pair {} from S3 {} bucket.".format(pair, args.bucket))
		download_folder_from_s3(args.bucket, pair, args.save_directory, args.access_key_id, args.secret_access_key)

	logging.info("Finished downloading all data.")

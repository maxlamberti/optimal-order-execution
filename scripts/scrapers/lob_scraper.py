import time
import logging
import krakenex
import argparse
import sentry_sdk
import pandas as pd
from requests.exceptions import HTTPError


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s', level=logging.INFO)


def upload_order_book_to_s3(order_book, s3_url):
    """Write data to S3 bucket in parquet format."""

    df = pd.DataFrame(order_book, columns=['price', 'volume', 'timestamp', 'type', 'query_time'])
    df['price'] = pd.to_numeric(df['price'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['type'] = df['type'].astype('category')
    df['query_time'] = pd.to_numeric(df['query_time'])

    logging.info("Uploading %s rows of order book data to S3 with url %s.", df.shape[0], s3_url)
    df.to_parquet(s3_url, compression='gzip')
    logging.info("Completed S3 upload.")


def get_order_book_data(pair, num_orders):
    """Fetch order book data for num_orders on ask and bid side of a currency pair."""

    try:
        logging.info("Querying kraken exchange for order book data for pair %s and 2*%s limit orders.", pair, num_orders)
        response = kraken.query_public('Depth', {'pair': pair, 'count': num_orders})
    except HTTPError as e:
        logging.error("Failed to query order book at Kraken. Error:\n%s", str(e))
        response = {}  # empty response

    if len(response.get('error', [])) > 0:
        logging.error("Error returned in response from Kraken:\n%s", response['error'])

    order_book = response.get('result', {}).get(pair, {'asks': [], 'bids': []})

    return order_book


def reformat_order_book_data(order_book, query_time):
    """Reformat order book data from original request in dict to tabular format."""

    asks = [row + ['a', query_time] for row in order_book['asks']]
    bids = [row + ['b', query_time] for row in order_book['bids']]
    order_book = asks + bids

    return order_book


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--pair',
        type=str,
        default='XXBTZUSD',
        required=False,
        help='Currency pair identifier for order book on kraken exchange.'
    )
    parser.add_argument(
        '-n',
        '--num_orders',
        type=int,
        default=10,
        required=False,
        help='Number of trades to fetch on each side of the limit order book.'
    )
    parser.add_argument(
        '-b',
        '--bucket',
        type=str,
        required=True,
        help='Address of S3 bucket to dump scraped data into.'
    )
    parser.add_argument(
        '-s',
        '--sentry',
        type=str,
        required=False,
        help='URL to Sentry project. Required for alert monitoring.'
    )
    parser.add_argument(
        '-q',
        '--query_period',
        type=float,
        default=1.0,
        required=False,
        help='Sets the periodicity of order book queries in seconds.'
    )
    parser.add_argument(
        '-w',
        '--write_period',
        type=float,
        default=3600,
        required=False,
        help='Sets the write periodicity to S3 in seconds.'
    )
    args = parser.parse_args()

    if args.sentry:
        sentry_sdk.init(args.sentry)  # handles error alerting via email

    kraken = krakenex.API()  # starts kraken session

    collection_cycle_start_time = time.time()  # tracking for upload periodicity to S3
    order_book = []  # initialize order book data tracker

    while True:

        # query order book
        query_time = time.time()
        new_order_book = get_order_book_data(args.pair, args.num_orders)
        order_book += reformat_order_book_data(new_order_book, query_time)

        # upload to S3 once significant amount of data is collected
        if (time.time() - collection_cycle_start_time) > args.write_period:
            s3_url = 's3://{}/{}/{}.parquet.gzip'.format(args.bucket, args.pair, int(time.time()))
            upload_order_book_to_s3(order_book, s3_url)
            collection_cycle_start_time = time.time()
            order_book = []

        # pause to keep in sync with desired query periodicity
        sleep_time = args.query_period - (time.time() - query_time)
        if sleep_time < 0:
            logging.warning("Order book query is taking %s seconds longer than desired.", abs(sleep_time))
        time.sleep(max(sleep_time, 0))
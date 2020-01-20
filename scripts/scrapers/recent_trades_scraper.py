import time
import logging
import krakenex
import argparse
import sentry_sdk
import pandas as pd
from requests.exceptions import HTTPError


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s', level=logging.INFO)


def get_recent_trade_data(pair, since):
    """Fetch recent trade data from Kraken."""

    try:
        logging.info("Querying kraken exchange for trade data for pair %s since time id %s.", pair, since)
        response = kraken.query_public('Trades', {'pair': pair, 'since': since})
    except HTTPError as e:
        logging.error("Failed to query trade data from Kraken. Error:\n%s", str(e))
        response = {}  # empty response

    if len(response.get('error', [])) > 0:
        logging.error("Error returned in response from Kraken:\n%s", response['error'])

    trades = response.get('result', {}).get(pair, [])
    last = response.get('result', {}).get('last', since)
    logging.info("Got data for %s trades from Kraken for pair %s", len(trades), pair)

    if len(trades) == 0:
        last = since  # dont update time stamp if no trades occured

    return trades, last


def get_kraken_server_time():
    """Fetch current server time at Kraken."""

    try:
        logging.info("Querying Kraken server time.")
        response = kraken.query_public('Time')
    except HTTPError as e:
        logging.error("Failed to get time from Kraken server. Error:\n%s", str(e))
        response = {}  # empty response

    if len(response.get('error', [])) > 0:
        logging.error("Error returned in response from Kraken:\n%s", response['error'])

    server_time = response.get('result', {}).get('unixtime', time.time())
    logging.info("Fetched kraken server time: %s", server_time)

    return server_time


def upload_trades_to_s3(trades, s3_url):
    """Write data to S3 bucket in parquet format."""

    df = pd.DataFrame(trades, columns=['price', 'volume', 'time', 'buy_or_sell', 'market_or_limit', 'misc'])
    del df['misc']  # remove
    df['price'] = pd.to_numeric(df['price'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['time'] = pd.to_numeric(df['time'])
    df['buy_or_sell'] = df['buy_or_sell'].astype('category')
    df['market_or_limit'] = df['market_or_limit'].astype('category')

    if df.shape[0] > 0:
        logging.info("Uploading %s trades to S3 with url %s.", df.shape[0], s3_url)
        df.to_parquet(s3_url, compression='gzip')
        logging.info("Completed S3 upload.")
    else:
        logging.info("No observed trades. Skipping upload to S3")


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
        '-t',
        '--time_id',
        type=float,
        default=-1,
        required=False,
        help='The time stamp id from which to start getting orders from kraken.'
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

    trade_ledger = []
    time_id = args.time_id  # used as initial time stamp for kraken to get data from
    collection_cycle_start_time = time.time()  # tracking for upload periodicity to S3

    if time_id == -1:  # set time id to current time
        time_id = get_kraken_server_time() * 10 ** 9

    while True:

        # query trades
        query_time = time.time()
        new_trades, time_id = get_recent_trade_data(args.pair, time_id)
        trade_ledger += new_trades
        logging.info("Current number of collected trades in memory: %s", len(trade_ledger))

        # upload to S3 once significant amount of data is collected
        if (time.time() - collection_cycle_start_time) > args.write_period:
            s3_url = 's3://{}/{}/{}.parquet.gzip'.format(args.bucket, args.pair, int(time.time()))
            upload_trades_to_s3(trade_ledger, s3_url)
            collection_cycle_start_time = time.time()
            trade_ledger = []

        # pause to keep in sync with desired query periodicity
        sleep_time = args.query_period - (time.time() - query_time)
        if sleep_time < 0:
            logging.warning("Trade query loop is taking %s seconds longer than desired.", abs(sleep_time))
        time.sleep(max(sleep_time, 0))

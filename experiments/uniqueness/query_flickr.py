"""
Simple utility to download images from Flickr based on a text query.
Requires user-specified API key.
Get a key at <https://www.flickr.com/services/apps/create/> for free.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com
"""
from itertools import takewhile
import os.path

import argparse
import flickrapi
import eta.core.storage as etas
import eta.core.utils as etau

DEFAULT_ROOT="data"

def main(args):

    # Flickr api access key
    flickr=flickrapi.FlickrAPI(args.key, args.secret, cache=True)

    # could also query by tags and tag_mode='all'
    photos = flickr.walk(text=args.query,
                         extras='url_c',
                         per_page=50,
                         sort='relevance')

    urls = []
    for photo in takewhile(lambda _: len(urls) < args.number, photos):
        url = photo.get('url_c')
        if url is not None:
            urls.append(url)

    if args.query_in_path:
        basedir = os.path.join(args.path, args.query)
    else:
        basedir = args.path
    etau.ensure_dir(basedir)

    client = etas.HTTPStorageClient()
    for url in urls:
        basename = client.get_filename(url)
        path_write = os.path.join(basedir, basename)

        client.download(url, path_write)
        print("Downloading to '%s'" % path_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("key", type=str, help="Flickr API key")
    parser.add_argument("secret", type=str, help="Secret to Flickr API key")
    parser.add_argument("query", type=str, help="Query string to use")
    parser.add_argument("-n", "--number", type=int, default=50,
                        help="number of images to download (default: 50)")
    parser.add_argument("-p", "--path", type=str, default=DEFAULT_ROOT,
                        help="path to download the images (created if needed)")
    parser.add_argument(
        "--query-in-path", "-i",
        dest="query_in_path",
        action="store_true")
    parser.add_argument(
        "--no-query-in-path",
        dest="query_in_path",
        action="store_false")
    parser.set_defaults(query_in_path=True)

    args = parser.parse_args()

    print(args.__dict__)
    main(args)

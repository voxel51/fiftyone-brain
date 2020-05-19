# FiftyOne Brain -- Experiment:  Uniqueness

How can we demonstrate that fiftyone can be used to rank samples in an otherwise unlabeled (or labeled) dataset by their "uniqueness"?

Problem: Unlabeled


## Full-Pass Example

The first thing in Uniqueness is a full-pass example through a small test-case.  This will also serve as the material for the walkthrough in the public fiftyone.

@todo fill this out

### Prepare The Data

This walkthrough will process a directory of images and compute their uniqueness.  The first thing we need to do is get some images.  
Let's get some images from Flickr, to keep this interesting!

You need a Flickr API key to do this.  If you already have a Flickr API key, then skip the next steps.  
1. Go to <https://www.flickr.com/services/apps/create/> 
2. Click on Request API Key. (<https://www.flickr.com/services/apps/create/apply/>) You will need to login (create account if needed, free).
3. Click on "Non-Commercial API Key" (this is just for a test usage); fill in the information on the next page.  You do not need to be very descriptive; your API will automatically appear on the following page.

Assume your key is `<KEY>` and its secret is `<SECRET>`.

You will also need to `pip install flickrapi` for this to work.

Next, let's download three sets of images to process together.  I suggest using three distinct object-nouns like "badger", "wolverine", and "kitten".  For the actual downloading, we will use the provided `query_flickr.py` script:

```
python query-flickr.py <KEY> <SECRET> "badger" 
python query-flickr.py <KEY> <SECRET> "wolverine" 
python query-flickr.py <KEY> <SECRET> "kitten" 
```


For the remainder of this walkthrough, let's assume the images are in directory `data`.

## Copyright

Copyright 2017-2020, Voxel51, Inc.<br>
voxel51.com

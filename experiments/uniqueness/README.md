# FiftyOne Brain -- Experiment:  Uniqueness

How can we demonstrate that fiftyone can be used to rank samples in an otherwise unlabeled (or labeled) dataset by their "uniqueness"?

Problem: Unlabeled


## Full-Pass Example

@todo move text between dashes to a FiftyOne walkthrough.

For a user-oriented example of uniqueness, please see the `dataset analysis` walkthrough in Fiftyone.

-----
In this walkthrough, we explore how uniqueness can be used to analyze a raw dataset.  We will work explore multiple ways of such analysis and use two datasets in the process.

# Part 1: Finding Duplicates and Near-Duplicates

A common problem in dataset creation is duplicated data.  Although this could be found using file-hashing---as in the `file_hashing` walkthrough---it is less possible when small manipulations have occurred in the data.  Even more critical for workflows involving model training is the need to get as much power out of each data samples as possible; near-duplicates, which are samples that are exceptionally similar to one another, are intrinsically less valuable for the training scenario.  Let's see if we can find such duplicates and near-duplicates in a common dataset: CIFAR-10.

## Load the Dataset

Open an IPython shell to begin.  We will use the CIFAR-10 dataset, which is available in the FiftyOne zoo.

```py
import fiftyone as fo
import fiftyone.zoo as foz

# Load the test split (automatically download if needed)
dataset = foz.load_zoo_dataset("cifar10", split="test")
```

## Process the Dataset

Now we can process the entire dataset for uniqueness.  This is a fairly expensive operation, but should finish in a few minutes at most.  We are processing through all samples in the dataset, then building a representation that relates the samples to each other.  Finally, we analyze this representation to output uniqueness.  

```py
import fiftyone.brain as fob

fob.compute_uniqueness(dataset)
```

The actual uniqueness values are associated with the samples in the dataset through what FiftyOne calls Insights.  Not that they are represented as a "keyed" group with each sample.

```py
sample = next(iter(dataset.default_view()))
print(sample)
print(sample.get_insights()['uniqueness'])
```

## Visualize the Output to Find Duplicates and Near-Duplicates

Now, let's visually inspect the output to see if we are able to identify 

```py
session = fo.launch_dashboard()
session.dataset = dataset

dups = dataset.default_view().sort_by("insights.uniqueness.scalar")
session.view = dups
```

You will easily see some near-duplicates in the GUI.  It surprised us that there are duplicates in CIFAR-10, too!

Of course, in this scenario, near duplicates are identified from visual inspection.  So, how do we get the information out of FiftyOne and back into your working environment.  Easy!  The `session` variable provides a bidirectional bridge between the GUI and the Python environment.  In this case, we will use the `session.selected` bridge.  So, in the GUI, click on the checkmark in the upper-left of some of the duplicates and near-duplicates.  Then, execute the following code in the IPython shell.

```py

dup_ids = session.selected

for id in dup_ids:
    dataset[id].add_tag("dup")

dups = dataset.default_view().match_tag("dup")
session.view = dups
```

And the GUI will only show these samples now.  We can, of course access the file-paths and other information about these samples programmatically so you can act on the findings.  But, let's do that at the end of Part 2 below!


# Part 2: Finding Unique Samples 

When building a dataset, it is important to create a diverse dataset with unique and representative samples.  Here, we explore FiftyOne's ability to help identify the most unique samples in a raw dataset.

## Prepare The Data

This walkthrough will process a directory of images and compute their uniqueness.  The first thing we need to do is get some images.  
Let's get some images from Flickr, to keep this interesting!

You need a Flickr API key to do this.  If you already have a Flickr API key, then skip the next steps.  
1. Go to <https://www.flickr.com/services/apps/create/> 
2. Click on Request API Key. (<https://www.flickr.com/services/apps/create/apply/>) You will need to login (create account if needed, free).
3. Click on "Non-Commercial API Key" (this is just for a test usage); fill in the information on the next page.  You do not need to be very descriptive; your API will automatically appear on the following page.

Assume your key is `<KEY>` and its secret is `<SECRET>`.

You will also need to `pip install flickrapi` for this to work.

Next, let's download three sets of images to process together.  I suggest using three distinct object-nouns like "badger", "wolverine", and "kitten".  For the actual downloading, we will use the provided `query_flickr.py` script:

```sh
python query-flickr.py <KEY> <SECRET> "badger" 
python query-flickr.py <KEY> <SECRET> "wolverine" 
python query-flickr.py <KEY> <SECRET> "kitten" 
```

For the remainder of this walkthrough, let's assume the images are in directory `data`.

## Load The Data Into FiftyOne

In an IPython shell, let's now work through getting this data into FiftyOne and working with it.

```py
import fiftyone as fo

dataset = fo.Dataset.from_images_dir('data', recursive=True, name='walkthrough')
```
This command uses a factory method on the `Dataset` class to traverse a directory of images (including subdirectories) and generate a dataset instance in FiftyOne.  This does not load the images from disk.  The first argument is the path to the images, the second is whether we should traverse recursively (yes in our case), and the third is a name for the dataset.

Now we can visualize it and get summary information about it quickly.
```py
print(dataset.summary())
session = fo.launch_dashboard()
session.dataset = dataset
```

Please refer to the `fifteen_to_fiftyone` and other walkthroughs for more 
useful things you can do with the dataset and dashboard.

## Compute Uniqueness and Analyze

Now, let's analyze the data.  For example, we may want to understand what are the most unique images among the data as they may inform or harm model training; we may want to discover duplicates or redundant samples.

Continuing in the same IPython session, let's compute and visualize uniqueness.

```py
import fiftyone.brain as fob

fob.compute_uniqueness(dataset)

# Now we have an insight group "uniqueness"
print(dataset.summary())

# Sort the visualization in the dashboard by most unique to least unique
session.view = dataset.default_view().sort_by("insights.uniqueness.scalar", reverse=True) 
```

Now, just visualizing the samples is interesting, but we want more.  We want to get the most unique samples from our dataset so that we can use them in our work.  Let's do just that.  In the same IPython session, execute the following code.

```py
# Get a view into the dataset that ranks it according to uniqueness
rank_view = dataset.default_view().sort_by("insights.uniqueness.scalar", reverse=True)

# Check the first one (should have a maximal uniqueness of 1.0)
sample = next(iter(rank_view))
print(sample)

# Generate a Python list with the 10 most unique samples
tenbest = [(x.id, x.filepath) for x in rank_view.limit(10)]

# Then you can do what you want with these.
# Output to csv or json, send images to your annotation team, seek additional similar data, etc.
```
-----

## Copyright

Copyright 2017-2020, Voxel51, Inc.<br>
voxel51.com

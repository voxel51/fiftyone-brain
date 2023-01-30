# FiftyOne Brain: Production

The tools needed for any backend production-oriented work in the brains behind
[FiftyOne](https://github.com/voxel51/fiftyone).

<img src="https://user-images.githubusercontent.com/3719547/74191434-8fe4f500-4c21-11ea-8d73-555edfce0854.png" alt="voxel51-logo.png" width="40%"/>

## Organization

-   `models/` code related to training/developing production models
-   `marketing/` scripts to generate reproducible marketing material

## Production models

The `models/` folder tracks the production models used by the FiftyOne Brain.
Brain models follow the ETA model development and deployment practices. For
full documentation,
[see here](https://github.com/voxel51/eta/blob/develop/docs/models_dev_guide.md).

### Adding a model to the Brain

To add a model to the Brain, following these steps:

1. In `fiftyone/brain/internal/models/cache/manifest.json`, add an entry for
   the model you want to publish (with info from steps below)

2. Upload the model's data blob to
   [this folder](https://drive.google.com/drive/u/1/folders/15lu2orhqGocHHgkprcye1gNXrFk2wrW0),
   turn on public link sharing, and put the file ID in the manifest entry

3. For the `manager.type` field of the manifest you can directly use
   `eta.core.models.ETAModelManager`. This is a class that knows how to
   download publicly available files from Google Drive

4. The `default_deployment_config_dict` is optional: it's for when you've
   implemented an `eta.core.learning.Model` subclass that knows how to load
   your model

> TODO Add Developer information with some example information.

## Copyright

Copyright 2017-2023, Voxel51, Inc.<br> voxel51.com

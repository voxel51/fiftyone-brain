# FiftyOne Brain: Production/Models/Simple_Resnet

Code for training the production `simple_resnet` model.

-   Training scripts
-   Dataset support is only CIFAR-10 now
-   Deployment script and documentation

## CIFAR-10 Classifier

The FiftyOne Brain Model Drive (BMD) is located at
[Shared drives://models/fiftyone-brain](https://drive.google.com/drive/u/1/folders/15lu2orhqGocHHgkprcye1gNXrFk2wrW0)
`15lu2orhqGocHHgkprcye1gNXrFk2wrW0`.

The name of the model weights file is `simple_resnet_cifar10.pth`.

Steps to train and deploy the model:

1. Run the `train_classifier_cifar10.bash` script

2. Upload the output `simple_resnet_cifar10.pth` to the BMD

3. Update the entry for the model in models manifest at
   `fiftyone/brain/internal/models/cache/manifest.json`. Be sure to include any
   information about the parameters stored there such as `image_mean` if they
   have changed. This is critical

## Copyright

Copyright 2017-2023, Voxel51, Inc.<br> voxel51.com

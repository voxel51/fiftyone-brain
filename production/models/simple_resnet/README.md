# FiftyOne Brain: Production/Models/Simple_Resnet

Code for training the production simple_resnet model.

-   Training scripts
-   Dataset support is only CIFAR-10 now
-   Deployment script and documentation

## CIFAR-10 Classifier

The FiftyOne Brain Model Drive (BMD) has id:
`15lu2orhqGocHHgkprcye1gNXrFk2wrW0`. It is at
`Shared Drives://models/fiftyone-brain`. The name of the model weights file is
`simple_resnet_cifar10.pth`.

Steps to train and deploy the model:

1. `bash ./train_classifier_cifar10.bash` from within your Voxel51 virtualenv.
2. Upload the output `simple_resnet_cifar10.pth` to the BMD.
3. Update the entry in the FiftyOne Brain model cache manifest. In the repo,
   this file is at `fiftyone/brain/models/cache/manifest.json`. Be sure to
   include any information about the parameters stored there such as
   `image_mean` if they have changed. This is critical.

## Copyright

Copyright 2017-2020, Voxel51, Inc.<br> voxel51.com

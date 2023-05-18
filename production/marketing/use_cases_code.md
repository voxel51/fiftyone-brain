# FiftyOne Use Cases

FiftyOne's data model is highly customizable. You can create datasets with
common content such as ground truth annotations, and you can freely add custom
fields to store additional information such as model predictions, metadata, and
more.

For example:

```py
import fiftyone as fo

# Create a basic test sample with ground truth annotation
sample = fo.Sample(
    filepath="/path/to/highway.jpg",
    tags=["test"],
    ground_truth=fo.Classification(label="highway"),
)

# Add a model prediction
sample["my_model"] = fo.Classification(label="highway", confidence=0.957)

# Add other custom metadata
sample.tags.append("difficult")
sample["frame_quaility"] = 62.9834
sample["weather"] = "sunny"

# Add sample to a dataset
dataset = fo.Dataset("highways")
dataset.add_sample(sample)

print(dataset.summary())
print(sample)
```

produces a dataset with the following content:

```
Name:           highways
Num samples:    1
Tags:           ['test', 'difficult']
Sample fields:
    filepath:       fiftyone.core.fields.StringField
    tags:           fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:       fiftyone.core.metadata.Metadata
    ground_truth:   fiftyone.core.labels.Classification
    my_model:       fiftyone.core.labels.Classification
    frame_quaility: fiftyone.core.fields.FloatField
    weather:        fiftyone.core.fields.StringField

{
    "_id": {
        "$oid": "5ecdd7f58efd8c18fcea4073"
    },
    "filepath": "/path/to/highway.jpg",
    "tags": [
        "test",
        "difficult"
    ],
    "metadata": null,
    "ground_truth": {
        "_cls": "Classification",
        "label": "highway",
        "confidence": null,
        "logits": null
    },
    "my_model": {
        "_cls": "Classification",
        "label": "highway",
        "confidence": 0.957,
        "logits": null
    },
    "frame_quaility": 62.9834,
    "weather": "sunny"
}
```

## Recommending next samples for annotation

FiftyOne provides builtin support for finding the most representative unlabeled
images in your dataset to send for annotation:

```py
import fiftyone.brain as fob

unlabeled_view = dataset.view().match_tag("unlabeled")

# Index your dataset by uniqueness
fob.compute_uniqueness(unlabeled_view)

# Recommend 10 most representative images to annotate
anno_view = unlabeled_view.sort_by("uniqueness", reverse=True).limit(10)

# Explore the samples visually in the GUI
fo.launch_dashboard(view=anno_view)
```

## Finding annotation errors

FiftyOne provides builtin support for finding likely annotation errors in your
dataset, using your model predictions to automatically inform the analysis:

```py
import fiftyone.brain as fob

# Index your dataset by likelihood of annotation error
fob.compute_mistakenness(dataset, pred_field="my_model")

# Get 10 most likely annotation mistakes
mistakes_view = dataset.view().sort_by("mistakenness", reverse=True).limit(10)

# Explore the samples visually in the GUI
fo.launch_dashboard(view=mistakes_view)
```

## Copyright

Copyright 2017-2023, Voxel51, Inc.<br> voxel51.com
